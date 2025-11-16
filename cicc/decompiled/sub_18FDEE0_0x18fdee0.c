// Function: sub_18FDEE0
// Address: 0x18fdee0
//
unsigned __int64 __fastcall sub_18FDEE0(__int64 a1)
{
  int v2; // ecx
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rsi
  int v5; // ecx
  __int64 v7; // rdx
  __int64 v8; // rsi
  int v9; // edi
  unsigned int v10; // edi
  _QWORD *v11; // rax
  unsigned __int64 v12; // rax
  int v13; // edx
  int v14; // r8d
  int v15; // edx
  unsigned __int64 v16; // rax
  int v17; // eax
  int v18; // eax
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  __int64 *v21; // rdi
  __int64 *v22; // rsi
  int v23; // [rsp+0h] [rbp-60h] BYREF
  int v24; // [rsp+4h] [rbp-5Ch] BYREF
  unsigned __int64 v25; // [rsp+8h] [rbp-58h] BYREF
  unsigned __int64 v26; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v27; // [rsp+18h] [rbp-48h] BYREF
  unsigned __int64 v28; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 v29; // [rsp+28h] [rbp-38h] BYREF
  unsigned __int64 v30; // [rsp+30h] [rbp-30h] BYREF
  int v31; // [rsp+38h] [rbp-28h]

  v2 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned int)(v2 - 35) > 0x11 )
  {
    if ( (unsigned __int8)(v2 - 75) > 1u )
    {
      v12 = sub_14B2890(a1, (__int64 *)&v25, (__int64 *)&v26, 0, 0);
      v14 = v13;
      v30 = v12;
      v15 = v12;
      v23 = v12;
      v31 = v14;
      if ( (unsigned int)(v12 - 1) > 3 )
      {
        v17 = *(unsigned __int8 *)(a1 + 16);
        if ( (unsigned int)(v15 - 7) > 1 )
        {
          if ( (unsigned int)(v17 - 60) > 0xC )
          {
            if ( (_BYTE)v17 == 86 )
            {
              v28 = sub_1597510(*(__int64 **)(a1 + 56), *(_QWORD *)(a1 + 56) + 4LL * *(unsigned int *)(a1 + 64));
              v29 = *(_QWORD *)(a1 - 24);
              LODWORD(v27) = *(unsigned __int8 *)(a1 + 16) - 24;
              return sub_18FD910((int *)&v27, (__int64 *)&v29, (__int64 *)&v28);
            }
            else if ( (_BYTE)v17 == 87 )
            {
              v27 = sub_1597510(*(__int64 **)(a1 + 56), *(_QWORD *)(a1 + 56) + 4LL * *(unsigned int *)(a1 + 64));
              v29 = *(_QWORD *)(a1 - 24);
              v28 = *(_QWORD *)(a1 - 48);
              v24 = *(unsigned __int8 *)(a1 + 16) - 24;
              return sub_18FD9D0(&v24, (__int64 *)&v28, (__int64 *)&v29, (__int64 *)&v27);
            }
            else
            {
              v20 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
              if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
              {
                v21 = *(__int64 **)(a1 - 8);
                v22 = &v21[v20];
              }
              else
              {
                v22 = (__int64 *)a1;
                v21 = (__int64 *)(a1 - v20 * 8);
              }
              v29 = sub_18FDB50(v21, v22);
              LODWORD(v28) = *(unsigned __int8 *)(a1 + 16) - 24;
              return sub_18FDAA0((int *)&v28, (__int64 *)&v29);
            }
          }
          else
          {
            v19 = *(_QWORD *)(a1 - 24);
            LODWORD(v27) = v17 - 24;
            v29 = v19;
            v28 = *(_QWORD *)a1;
            return sub_18FD850((int *)&v27, (__int64 *)&v28, (__int64 *)&v29);
          }
        }
      }
      else
      {
        v16 = v25;
        if ( v25 > v26 )
        {
          v25 = v26;
          v26 = v16;
        }
        v17 = *(unsigned __int8 *)(a1 + 16);
      }
      LODWORD(v29) = v17 - 24;
      return sub_18FD780(&v29, &v23, &v25, &v26);
    }
    v7 = *(_QWORD *)(a1 - 48);
    v8 = *(_QWORD *)(a1 - 24);
    v9 = *(unsigned __int16 *)(a1 + 18);
    v29 = v7;
    v10 = v9 & 0xFFFF7FFF;
    v30 = v8;
    LODWORD(v27) = v10;
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v11 = *(_QWORD **)(a1 - 8);
    else
      v11 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( v11[3] < *v11 )
    {
      v29 = v8;
      v30 = v7;
      v18 = sub_15FF5D0(v10);
      v2 = *(unsigned __int8 *)(a1 + 16);
      LODWORD(v27) = v18;
    }
    LODWORD(v28) = v2 - 24;
    return sub_18FD6B0(&v28, (int *)&v27, &v29, &v30);
  }
  else
  {
    v3 = *(_QWORD *)(a1 - 48);
    v4 = *(_QWORD *)(a1 - 24);
    v5 = v2 - 24;
    v29 = v3;
    v30 = v4;
    if ( ((1LL << v5) & 0x1C019800) != 0 && v3 > v4 )
    {
      v29 = v4;
      v30 = v3;
    }
    LODWORD(v28) = v5;
    return sub_18FD5F0((int *)&v28, (__int64 *)&v29, (__int64 *)&v30);
  }
}
