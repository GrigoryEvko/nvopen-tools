// Function: sub_6BEBB0
// Address: 0x6bebb0
//
__int64 __fastcall sub_6BEBB0(__int64 a1, _QWORD *a2, unsigned int a3, _DWORD *a4, _DWORD *a5, _DWORD *a6)
{
  int v8; // r14d
  int v9; // r10d
  int v10; // eax
  __int64 result; // rax
  unsigned int v12; // r14d
  int v13; // r14d
  __int16 v14; // ax
  int v15; // eax
  char v16; // cl
  __int64 v17; // rax
  char i; // dl
  __int64 v19; // rax
  int v20; // r8d
  _QWORD *v21; // r14
  int v22; // eax
  int v23; // eax
  int v24; // [rsp+8h] [rbp-C8h]
  __int64 v26; // [rsp+10h] [rbp-C0h]
  int v27; // [rsp+18h] [rbp-B8h]
  char v29; // [rsp+24h] [rbp-ACh] BYREF
  _BYTE v30[4]; // [rsp+28h] [rbp-A8h] BYREF
  int v31; // [rsp+2Ch] [rbp-A4h] BYREF
  unsigned int v32; // [rsp+30h] [rbp-A0h] BYREF
  int v33; // [rsp+34h] [rbp-9Ch] BYREF
  __int64 v34; // [rsp+38h] [rbp-98h] BYREF
  _OWORD v35[3]; // [rsp+40h] [rbp-90h] BYREF
  _BYTE v36[96]; // [rsp+70h] [rbp-60h] BYREF

  *a5 = 0;
  *a4 = 0;
  v8 = sub_8D32E0(a1);
  v9 = sub_8D3110(a1);
  v10 = 16778264;
  if ( a3 != 2 )
    v10 = 16778248;
  v27 = v10;
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u || (result = word_4D04898) != 0 )
  {
    v24 = v9;
    if ( !v8 )
    {
      v12 = sub_8D2600(a1);
      if ( v12 )
      {
LABEL_7:
        result = (unsigned int)*a5;
        goto LABEL_8;
      }
      v32 = 0;
      v33 = 0;
      if ( (unsigned int)sub_8407C0((_DWORD)a2, a1, 0, 0, 0, 0, 0, v27, (__int64)v35, (__int64)v36, (__int64)&v33) )
      {
        if ( !unk_4D04460 || (v35[1] & 2) == 0 )
          v12 = qword_4D0495C == 0;
        LOBYTE(v14) = v35[1] & 0xFB;
        HIBYTE(v14) = ((unsigned __int16)(v35[1] & 0xFDFB) >> 8) | 2;
        LOWORD(v35[1]) = v14;
        sub_8449E0(a2, a1, v35, v36, v12);
        goto LABEL_16;
      }
      if ( dword_4D0478C && (unsigned int)sub_8D3BB0(a1) )
      {
        v34 = sub_6E3060(a2);
        sub_6BE930(a1, 0, 1u, &v34, &v32);
        if ( v32 )
        {
          sub_832E80(v34);
          sub_6BD7E0(a1, a3, v34, (__int64)a2);
          v16 = *((_BYTE *)a2 + 16);
          if ( !v16 )
            goto LABEL_72;
          v17 = *a2;
          for ( i = *(_BYTE *)(*a2 + 140LL); i == 12; i = *(_BYTE *)(v17 + 140) )
            v17 = *(_QWORD *)(v17 + 160);
          if ( i )
          {
            if ( v16 == 1 )
            {
              *(_BYTE *)(a2[18] + 26LL) &= ~1u;
              if ( a3 == 3 )
                *(_BYTE *)(a2[18] + 25LL) |= 0x40u;
            }
            v19 = sub_68B740((__int64)a2);
            if ( v19 )
              *(_BYTE *)(v19 + 169) = *(_BYTE *)(v19 + 169) & 0x9F | 0x40;
          }
          else
          {
LABEL_72:
            *a6 = 1;
            sub_6E6960(312, a2, *a2, a1);
          }
          *a5 = 1;
        }
        else
        {
          sub_6E6960(312, a2, *a2, a1);
          v33 = 1;
        }
        sub_6E1990(v34);
      }
      result = (unsigned int)*a5;
      if ( v33 )
      {
        if ( !(_DWORD)result )
        {
          *a6 = 1;
          *a5 = 1;
        }
        goto LABEL_17;
      }
LABEL_8:
      if ( !(_DWORD)result )
        return result;
      goto LABEL_17;
    }
    v26 = sub_8D46C0(a1);
    v13 = sub_82EAE0();
    nullsub_6(a2);
    if ( (unsigned int)sub_831CF0(
                         (_DWORD)a2,
                         *a2,
                         a1,
                         1,
                         v27,
                         (unsigned int)&v29,
                         (__int64)v30,
                         (__int64)&v31,
                         (__int64)&v32,
                         (__int64)&v33,
                         (__int64)v36)
      || v32 && a3 - 1 <= 1 )
    {
      *a4 = v31;
      result = (unsigned int)*a5;
      goto LABEL_8;
    }
    v33 = 0;
    *a4 = v31;
    if ( (unsigned int)sub_8DD3B0(*a2) )
    {
      v33 = 1;
      goto LABEL_15;
    }
    if ( v13 && v24 && !(unsigned int)sub_6ED230(a2) )
      goto LABEL_7;
    if ( (unsigned int)sub_8E31E0(*a2) )
    {
      v20 = sub_8413E0((_DWORD)a2, a1, v27, 0, (unsigned int)v35, (unsigned int)&v34, 0);
      if ( (unsigned int)v34 | v20 )
      {
        if ( a3 != 3 && (_DWORD)v34 && (_DWORD)qword_4F077B4 )
          goto LABEL_7;
LABEL_37:
        if ( !v33 )
        {
          sub_842520(a2, a1, v35, 1, 0, 171);
          sub_6FAB30(a2, a1, 1, 0, 0);
          goto LABEL_16;
        }
LABEL_15:
        sub_6F4200(a2, a1, a3, 0);
LABEL_16:
        *a5 = 1;
LABEL_17:
        result = sub_6E26D0(2, a2);
        if ( *((_BYTE *)a2 + 16) == 1 )
        {
          result = a2[18];
          if ( *(_BYTE *)(result + 24) == 1 && *(_BYTE *)(result + 56) == 9 )
          {
            *(_BYTE *)(result + 27) &= ~2u;
            if ( a3 == 2 )
            {
              *(_BYTE *)(result + 25) |= 0x80u;
            }
            else if ( a3 == 3 )
            {
              *(_BYTE *)(result + 25) |= 0x40u;
            }
          }
        }
        return result;
      }
    }
    if ( !v31 )
      goto LABEL_7;
    if ( (unsigned int)sub_8D3A70(v26) )
    {
      if ( *((_BYTE *)a2 + 17) == 1 && (unsigned int)sub_8D3A70(*a2) && sub_8D5CE0(v26, *a2) )
        goto LABEL_7;
      v15 = sub_836C50((_DWORD)a2, 0, v26, 1, 1, 1, a1, 0, 0, (__int64)v35, 0, (__int64)&v34, 0);
      if ( !((unsigned int)v34 | v15) )
        goto LABEL_7;
    }
    else
    {
      if ( !(unsigned int)sub_8D3A70(*a2) )
      {
        v21 = 0;
        if ( *((_BYTE *)a2 + 16) == 2 )
          v21 = a2 + 18;
        memset(v35, 0, sizeof(v35));
        if ( v32 )
          goto LABEL_7;
        v22 = sub_6EB660(a2);
        if ( !(unsigned int)sub_8E1010(
                              *a2,
                              v21 != 0,
                              (*((_BYTE *)a2 + 19) & 0x10) != 0,
                              v22,
                              0,
                              (_DWORD)v21,
                              v26,
                              0,
                              0,
                              0,
                              0,
                              (__int64)&v35[1] + 8,
                              0) )
          goto LABEL_7;
        goto LABEL_37;
      }
      v23 = sub_840360((_DWORD)a2, v26, 0, 0, 1, 1, a1, 0, 0, (__int64)v35, (__int64)&v34, 0);
      if ( !((unsigned int)v34 | v23) )
        goto LABEL_7;
    }
    LOBYTE(v35[1]) &= ~4u;
    goto LABEL_37;
  }
  return result;
}
