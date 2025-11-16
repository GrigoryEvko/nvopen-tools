// Function: sub_BD4FF0
// Address: 0xbd4ff0
//
unsigned __int64 __fastcall sub_BD4FF0(unsigned __int8 *a1, __int64 a2, _BYTE *a3, _BYTE *a4)
{
  char v8; // al
  int v9; // eax
  unsigned __int64 v10; // r13
  __int64 v12; // rcx
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rdi
  int v18; // edx
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r13
  unsigned __int8 v23; // al
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int8 v27; // dl
  __int64 v28; // rax
  __int64 v29; // r14
  unsigned int v30; // r13d
  __int64 v31; // rax
  unsigned __int8 v32; // dl
  __int64 v33; // rax
  __int64 v34; // r14
  unsigned int v35; // r12d
  _QWORD v36[8]; // [rsp+0h] [rbp-40h] BYREF

  *a3 = 0;
  v8 = qword_4F83648[8];
  if ( LOBYTE(qword_4F83648[8]) )
    v8 = sub_BD4ED0((__int64)a1);
  *a4 = v8;
  v9 = *a1;
  if ( (_BYTE)v9 == 22 )
  {
    v10 = sub_B2BD50((__int64)a1);
    if ( !v10 )
    {
      if ( (v21 = sub_B2BCD0((__int64)a1), (v22 = v21) == 0)
        || (v23 = *(_BYTE *)(v21 + 8), v23 != 12)
        && v23 > 3u
        && v23 != 5
        && (v23 & 0xFB) != 0xA
        && (v23 & 0xFD) != 4
        && ((unsigned __int8)(v23 - 15) > 3u && v23 != 20 || !(unsigned __int8)sub_BCEBA0(v22, 0))
        || (v36[0] = sub_9208B0(a2, v22), v36[1] = v24, (v10 = (unsigned __int64)(v36[0] + 7LL) >> 3) == 0) )
      {
        v25 = sub_B2BD60((__int64)a1);
        *a3 = 1;
        return v25;
      }
    }
    return v10;
  }
  if ( (unsigned __int8)v9 <= 0x1Cu
    || (unsigned __int8)(v9 - 34) > 0x33u
    || (v12 = 0x8000000000041LL, !_bittest64(&v12, (unsigned int)(v9 - 34))) )
  {
    switch ( (_BYTE)v9 )
    {
      case 0x3D:
      case 0x4D:
        if ( (a1[7] & 0x20) != 0 )
        {
          v26 = sub_B91C10((__int64)a1, 12);
          if ( v26 )
          {
            v27 = *(_BYTE *)(v26 - 16);
            v28 = (v27 & 2) != 0 ? *(_QWORD *)(v26 - 32) : v26 - 8LL * ((v27 >> 2) & 0xF) - 16;
            v29 = *(_QWORD *)(*(_QWORD *)v28 + 136LL);
            v30 = *(_DWORD *)(v29 + 32);
            if ( v30 > 0x40 )
            {
              if ( v30 - (unsigned int)sub_C444A0(v29 + 24) > 0x40 )
                return -1;
              v10 = **(_QWORD **)(v29 + 24);
            }
            else
            {
              v10 = *(_QWORD *)(v29 + 24);
            }
            if ( v10 )
              return v10;
          }
          if ( (a1[7] & 0x20) != 0 )
          {
            v31 = sub_B91C10((__int64)a1, 13);
            if ( v31 )
            {
              v32 = *(_BYTE *)(v31 - 16);
              if ( (v32 & 2) != 0 )
                v33 = *(_QWORD *)(v31 - 32);
              else
                v33 = v31 - 8LL * ((v32 >> 2) & 0xF) - 16;
              v34 = *(_QWORD *)(*(_QWORD *)v33 + 136LL);
              v35 = *(_DWORD *)(v34 + 32);
              if ( v35 > 0x40 )
              {
                v10 = -1;
                if ( v35 - (unsigned int)sub_C444A0(v34 + 24) <= 0x40 )
                  v10 = **(_QWORD **)(v34 + 24);
              }
              else
              {
                v10 = *(_QWORD *)(v34 + 24);
              }
              goto LABEL_20;
            }
          }
        }
        v10 = 0;
LABEL_20:
        *a3 = 1;
        return v10;
      case 0x3C:
        if ( !(unsigned __int8)sub_B4CE70((__int64)a1) )
        {
          v19 = *((_QWORD *)a1 + 9);
          goto LABEL_29;
        }
        break;
      case 3:
        if ( (v17 = *((_QWORD *)a1 + 3), v18 = *(unsigned __int8 *)(v17 + 8), (_BYTE)v18 == 12)
          || (unsigned __int8)v18 <= 3u
          || (_BYTE)v18 == 5
          || (v18 & 0xFB) == 0xA
          || (v18 & 0xFD) == 4
          || ((unsigned __int8)(*(_BYTE *)(v17 + 8) - 15) <= 3u || v18 == 20) && (unsigned __int8)sub_BCEBA0(v17, 0) )
        {
          if ( (a1[32] & 0xF) != 9 )
          {
            v19 = *((_QWORD *)a1 + 3);
LABEL_29:
            v20 = sub_9208B0(a2, v19);
            *a3 = 0;
            *a4 = 0;
            return (unsigned __int64)(v20 + 7) >> 3;
          }
        }
        break;
    }
    return 0;
  }
  v10 = sub_A74620((_QWORD *)a1 + 9);
  v13 = *((_QWORD *)a1 - 4);
  if ( v13 )
  {
    if ( !*(_BYTE *)v13 && *(_QWORD *)(v13 + 24) == *((_QWORD *)a1 + 10) )
    {
      v36[0] = *(_QWORD *)(v13 + 120);
      v14 = sub_A74620(v36);
      if ( v10 < v14 )
        v10 = v14;
    }
  }
  if ( !v10 )
  {
    v10 = sub_A74640((_QWORD *)a1 + 9);
    v15 = *((_QWORD *)a1 - 4);
    if ( v15 )
    {
      if ( !*(_BYTE *)v15 && *(_QWORD *)(v15 + 24) == *((_QWORD *)a1 + 10) )
      {
        v36[0] = *(_QWORD *)(v15 + 120);
        v16 = sub_A74640(v36);
        if ( v10 < v16 )
          v10 = v16;
      }
    }
    goto LABEL_20;
  }
  return v10;
}
