// Function: sub_3374A50
// Address: 0x3374a50
//
__int64 __fastcall sub_3374A50(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r14
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // r9
  __int64 v8; // r15
  unsigned int v9; // r13d
  unsigned __int64 v10; // r12
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 v13; // r13
  unsigned int v14; // [rsp+Ch] [rbp-34h]

  result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 2416LL))(a1, a2, 0);
  if ( result )
  {
    v7 = *(unsigned int *)(a2 + 68);
    v8 = result;
    if ( (_DWORD)v7 == 1 )
    {
      v12 = *(unsigned int *)(a3 + 8);
      v13 = v6;
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v12 + 1, 0x10u, 0xFFFFFFFF00000000LL, v7);
        v12 = *(unsigned int *)(a3 + 8);
      }
      result = *(_QWORD *)a3 + 16 * v12;
      *(_QWORD *)result = v8;
      *(_QWORD *)(result + 8) = v13;
      ++*(_DWORD *)(a3 + 8);
    }
    else if ( (_DWORD)v7 )
    {
      result = *(unsigned int *)(a3 + 8);
      v9 = 0;
      do
      {
        v10 = v3 & 0xFFFFFFFF00000000LL | v9;
        v3 = v10;
        if ( result + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          v14 = v7;
          sub_C8D5F0(a3, (const void *)(a3 + 16), result + 1, 0x10u, 0xFFFFFFFF00000000LL, v7);
          result = *(unsigned int *)(a3 + 8);
          v7 = v14;
        }
        v11 = (__int64 *)(*(_QWORD *)a3 + 16 * result);
        ++v9;
        *v11 = v8;
        v11[1] = v10;
        result = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
        *(_DWORD *)(a3 + 8) = result;
      }
      while ( (_DWORD)v7 != v9 );
    }
  }
  return result;
}
