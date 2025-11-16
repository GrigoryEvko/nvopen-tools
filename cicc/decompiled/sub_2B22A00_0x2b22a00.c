// Function: sub_2B22A00
// Address: 0x2b22a00
//
__int64 __fastcall sub_2B22A00(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r12
  _QWORD *v10; // rax
  __int64 v11; // r10
  _QWORD **v12; // rdx
  int v13; // ecx
  __int64 *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rdx
  unsigned int v19; // esi
  __int64 v21; // [rsp+18h] [rbp-68h]
  _DWORD v22[8]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v23; // [rsp+40h] [rbp-40h]

  if ( a2 > 0xF )
  {
    v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 80) + 56LL))(*(_QWORD *)(a1 + 80));
    if ( !v8 )
    {
      v23 = 257;
      v10 = sub_BD2C40(72, unk_3F10FD0);
      v11 = a5;
      v8 = (__int64)v10;
      if ( v10 )
      {
        v12 = *(_QWORD ***)(a3 + 8);
        v13 = *((unsigned __int8 *)v12 + 8);
        if ( (unsigned int)(v13 - 17) > 1 )
        {
          v15 = sub_BCB2A0(*v12);
        }
        else
        {
          BYTE4(v21) = (_BYTE)v13 == 18;
          LODWORD(v21) = *((_DWORD *)v12 + 8);
          v14 = (__int64 *)sub_BCB2A0(*v12);
          v15 = sub_BCE1B0(v14, v21);
        }
        sub_B523C0(v8, v15, 53, a2, a3, a4, (__int64)v22, 0, 0, 0);
        v11 = a5;
      }
      (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
        *(_QWORD *)(a1 + 88),
        v8,
        v11,
        *(_QWORD *)(a1 + 56),
        *(_QWORD *)(a1 + 64));
      v16 = *(_QWORD *)a1;
      v17 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
      if ( *(_QWORD *)a1 != v17 )
      {
        do
        {
          v18 = *(_QWORD *)(v16 + 8);
          v19 = *(_DWORD *)v16;
          v16 += 16;
          sub_B99FD0(v8, v19, v18);
        }
        while ( v17 != v16 );
      }
    }
  }
  else
  {
    v22[1] = 0;
    return sub_B35C90(a1, a2, a3, a4, a5, a6, v22[0], 0);
  }
  return v8;
}
