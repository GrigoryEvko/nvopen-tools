// Function: sub_72DA30
// Address: 0x72da30
//
__int64 __fastcall sub_72DA30(__int64 a1, __int64 a2, char a3)
{
  __int64 result; // rax
  __int64 *v7; // rbx
  int v8; // edi
  int v9; // r13d
  _BYTE *v10; // r12
  __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-38h]

  result = *(unsigned int *)(*(_QWORD *)(a1 + 48) + 160LL);
  if ( (_DWORD)result )
  {
    v12 = *(_QWORD *)(a1 + 48);
    result = sub_72B840(v12);
    v7 = *(__int64 **)(result + 216);
    if ( v7 )
    {
      while ( *((_BYTE *)v7 + 16) != a3 || v7[4] != a2 )
      {
        v7 = (__int64 *)*v7;
        if ( !v7 )
          return result;
      }
      v8 = *(_DWORD *)(v12 + 164);
      v9 = dword_4F07270[0];
      if ( dword_4F07270[0] == v8 )
        v9 = 0;
      else
        sub_7296B0(v8);
      v10 = sub_7264B0();
      sub_729730(v9);
      *(_QWORD *)v10 = *v7;
      *v7 = (__int64)v10;
      v10[16] = a3;
      v11 = v7[1];
      *((_QWORD *)v10 + 4) = a1;
      *((_QWORD *)v10 + 1) = v11;
      result = *((unsigned __int8 *)v7 + 24);
      v10[24] = result;
    }
  }
  return result;
}
