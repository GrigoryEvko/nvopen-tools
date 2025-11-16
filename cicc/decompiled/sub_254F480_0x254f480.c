// Function: sub_254F480
// Address: 0x254f480
//
__int64 __fastcall sub_254F480(__int64 *a1, unsigned __int64 a2, _BYTE *a3)
{
  unsigned __int16 v6; // ax
  unsigned int v7; // r12d
  unsigned __int8 *v9; // r14
  int v10; // eax
  __int64 v11; // r9
  __int64 v12; // r13
  __int64 v13; // r15
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  _BYTE *v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _BYTE *v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // rbx
  __int64 v23; // rax
  unsigned __int8 *v24; // rdx
  unsigned __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-48h]
  char v29; // [rsp+17h] [rbp-39h] BYREF
  __int64 v30[7]; // [rsp+18h] [rbp-38h] BYREF

  sub_AE6EC0(*a1 + 104, a2);
  v6 = sub_D139D0((__int64 *)a2, 0, 0, 0);
  if ( !(_BYTE)v6 && HIBYTE(v6) || (v9 = *(unsigned __int8 **)(a2 + 24), v10 = *v9, (unsigned __int8)v10 <= 0x1Cu) )
  {
    *a3 = 1;
    return 1;
  }
  else
  {
    v11 = *a1;
    if ( (_BYTE)v10 == 82 )
    {
      if ( (v9[7] & 0x40) != 0 )
        v24 = (unsigned __int8 *)*((_QWORD *)v9 - 1);
      else
        v24 = &v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
      v7 = 1;
      if ( **(_BYTE **)&v9[32 * (a2 == (_QWORD)v24) - 64] > 0x15u )
      {
        v25 = *(_QWORD *)(v11 + 72) & 0xFFFFFFFFFFFFFFFCLL;
        if ( (*(_QWORD *)(v11 + 72) & 3LL) == 3 )
          v25 = *(_QWORD *)(v25 + 24);
        LOBYTE(v7) = *(_QWORD *)a2 == v25;
      }
    }
    else
    {
      v12 = a1[2];
      v13 = a1[1];
      if ( (_BYTE)v10 == 30 )
      {
        v26 = *(_QWORD *)(a2 + 24);
        v28 = *a1;
        v30[0] = v12;
        v29 = 0;
        v27 = sub_B43CB0(v26);
        return (unsigned int)sub_25230B0(
                               v13,
                               (__int64 (__fastcall *)(__int64, __int64 *))sub_253A970,
                               (__int64)v30,
                               v27,
                               1,
                               v28,
                               &v29,
                               0);
      }
      v14 = (unsigned int)(v10 - 34);
      if ( (unsigned __int8)v14 > 0x33u )
        return 0;
      v15 = 0x8000000000041LL;
      if ( !_bittest64(&v15, v14) )
        return 0;
      v7 = 1;
      if ( (unsigned __int8 *)a2 == v9 - 32 )
        return v7;
      if ( !sub_254C190(*(unsigned __int8 **)(a2 + 24), a2) )
        return 0;
      if ( **((_BYTE **)v9 - 4) )
        return 0;
      v16 = (_BYTE *)*((_QWORD *)v9 - 4);
      v7 = sub_254F400(v13, v16);
      if ( !(_BYTE)v7 )
        return 0;
      v20 = v16;
      v21 = (__int64)(a2 - (_QWORD)&v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)]) >> 5;
      if ( (v16[2] & 1) != 0 )
      {
        sub_B2C6D0((__int64)v16, (__int64)v16, (__int64)v16, v17);
        v20 = v16;
      }
      v22 = *((_QWORD *)v20 + 12) + 40LL * (unsigned int)v21;
      v23 = *(unsigned int *)(v12 + 8);
      if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
      {
        sub_C8D5F0(v12, (const void *)(v12 + 16), v23 + 1, 8u, v18, v19);
        v23 = *(unsigned int *)(v12 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v12 + 8 * v23) = v22;
      ++*(_DWORD *)(v12 + 8);
    }
  }
  return v7;
}
