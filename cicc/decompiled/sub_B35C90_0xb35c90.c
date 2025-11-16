// Function: sub_B35C90
// Address: 0xb35c90
//
__int64 __fastcall sub_B35C90(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        char a8)
{
  __int64 v12; // r13
  _QWORD *v13; // rdx
  int v14; // esi
  __int64 v15; // rax
  int v16; // eax
  unsigned int *v17; // rbx
  __int64 v18; // r12
  __int64 v19; // rdx
  __int64 v20; // rsi
  unsigned int v21; // [rsp+Ch] [rbp-84h]
  int v22; // [rsp+10h] [rbp-80h]
  __int64 v24; // [rsp+28h] [rbp-68h]
  _BYTE v25[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v26; // [rsp+50h] [rbp-40h]

  if ( *(_BYTE *)(a1 + 108) )
    return sub_B35B80(a1, 103 - ((unsigned int)(a8 == 0) - 1), a2, a3, a4, a5, 0);
  v22 = a4;
  v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 80) + 56LL))(*(_QWORD *)(a1 + 80));
  if ( !v12 )
  {
    v21 = *(_DWORD *)(a1 + 104);
    if ( BYTE4(a7) )
      v21 = a7;
    v26 = 257;
    v12 = sub_BD2C40(72, unk_3F10FD0);
    if ( v12 )
    {
      v13 = *(_QWORD **)(a3 + 8);
      v14 = *((unsigned __int8 *)v13 + 8);
      if ( (unsigned int)(v14 - 17) > 1 )
      {
        v16 = sub_BCB2A0(*v13);
      }
      else
      {
        BYTE4(v24) = (_BYTE)v14 == 18;
        LODWORD(v24) = *((_DWORD *)v13 + 8);
        v15 = sub_BCB2A0(*v13);
        v16 = sub_BCE1B0(v15, v24);
      }
      sub_B523C0(v12, v16, 54, a2, a3, v22, (__int64)v25, 0, 0, 0);
    }
    if ( a6 || (a6 = *(_QWORD *)(a1 + 96)) != 0 )
      sub_B99FD0(v12, 3, a6);
    sub_B45150(v12, v21);
    (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
      *(_QWORD *)(a1 + 88),
      v12,
      a5,
      *(_QWORD *)(a1 + 56),
      *(_QWORD *)(a1 + 64));
    v17 = *(unsigned int **)a1;
    v18 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v18 )
    {
      do
      {
        v19 = *((_QWORD *)v17 + 1);
        v20 = *v17;
        v17 += 4;
        sub_B99FD0(v12, v20, v19);
      }
      while ( (unsigned int *)v18 != v17 );
    }
  }
  return v12;
}
