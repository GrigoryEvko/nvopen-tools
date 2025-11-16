// Function: sub_30609A0
// Address: 0x30609a0
//
__int64 __fastcall sub_30609A0(__int64 *a1, __int64 *a2, unsigned __int64 *a3)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 result; // rax
  __int64 v8; // rbx
  _QWORD *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rdi
  char *v12; // rsi
  char v13; // bl
  __int64 v14; // rax
  __int64 v15; // rbx
  _QWORD *v16; // rax
  char v17; // [rsp+7h] [rbp-19h] BYREF
  _QWORD *v18; // [rsp+8h] [rbp-18h] BYREF

  v5 = a2[1];
  v6 = *a2;
  switch ( v5 )
  {
    case 15LL:
      if ( *(_QWORD *)v6 == 0x746E692D6D76766ELL
        && *(_DWORD *)(v6 + 8) == 1634872690
        && *(_WORD *)(v6 + 12) == 26478
        && *(_BYTE *)(v6 + 14) == 101 )
      {
        v11 = sub_22077B0(0x10u);
        if ( v11 )
          *(_QWORD *)v11 = &unk_4A30FE0;
        goto LABEL_20;
      }
      return 0;
    case 18LL:
      if ( !(*(_QWORD *)v6 ^ 0x706E656765646F63LL | *(_QWORD *)(v6 + 8) ^ 0x6373657261706572LL)
        && *(_WORD *)(v6 + 16) == 30309 )
      {
        v8 = *a1;
        v9 = (_QWORD *)sub_22077B0(0x10u);
        v10 = v8 + 2248;
        v11 = (__int64)v9;
        if ( v9 )
        {
          v9[1] = v10;
          *v9 = &unk_4A31020;
          v12 = (char *)a3[1];
          v18 = v9;
          if ( v12 != (char *)a3[2] )
          {
LABEL_21:
            if ( v12 )
            {
              *(_QWORD *)v12 = v11;
              a3[1] += 8LL;
              return 1;
            }
            a3[1] = 8;
LABEL_30:
            if ( v11 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
            return 1;
          }
LABEL_10:
          sub_2353750(a3, v12, &v18);
          v11 = (__int64)v18;
          goto LABEL_30;
        }
LABEL_20:
        v18 = (_QWORD *)v11;
        v12 = (char *)a3[1];
        if ( v12 != (char *)a3[2] )
          goto LABEL_21;
        goto LABEL_10;
      }
      return 0;
    case 17LL:
      if ( !(*(_QWORD *)v6 ^ 0x74732D7265776F6CLL | *(_QWORD *)(v6 + 8) ^ 0x6772612D74637572LL)
        && *(_BYTE *)(v6 + 16) == 115 )
      {
        v13 = *(_BYTE *)(*a1 + 539453);
        v14 = sub_22077B0(0x10u);
        v11 = v14;
        if ( v14 )
        {
          *(_BYTE *)(v14 + 8) = v13;
          *(_QWORD *)v14 = &unk_4A11DB8;
        }
        goto LABEL_20;
      }
      return 0;
    case 31LL:
      if ( !memcmp((const void *)v6, "nvptx-set-local-array-alignment", 0x1Fu) )
      {
        nullsub_1908(&v17);
        v11 = sub_22077B0(0x10u);
        if ( v11 )
          *(_QWORD *)v11 = &unk_4A31060;
        goto LABEL_20;
      }
      return 0;
    case 21LL:
      if ( !(*(_QWORD *)v6 ^ 0x6F632D787470766ELL | *(_QWORD *)(v6 + 8) ^ 0x6C617679622D7970LL)
        && *(_DWORD *)(v6 + 16) == 1735549229
        && *(_BYTE *)(v6 + 20) == 115 )
      {
        v11 = sub_22077B0(0x10u);
        if ( v11 )
          *(_QWORD *)v11 = &unk_4A310A0;
        goto LABEL_20;
      }
      return 0;
  }
  if ( v5 != 16 )
    return 0;
  result = 0;
  if ( !(*(_QWORD *)v6 ^ 0x6F6C2D787470766ELL | *(_QWORD *)(v6 + 8) ^ 0x736772612D726577LL) )
  {
    v15 = *a1;
    v16 = (_QWORD *)sub_22077B0(0x10u);
    v11 = (__int64)v16;
    if ( v16 )
    {
      v16[1] = v15;
      *v16 = &unk_4A310E0;
    }
    goto LABEL_20;
  }
  return result;
}
