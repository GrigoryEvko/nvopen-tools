// Function: sub_6F85E0
// Address: 0x6f85e0
//
__int64 __fastcall sub_6F85E0(__int64 *a1, __int64 a2, __int64 a3, _QWORD *a4, __m128i *a5, __int64 a6)
{
  unsigned int v7; // r14d
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  int v17; // r9d
  __int64 *v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rsi
  __int64 v25; // rdi
  __int64 v26; // [rsp-18h] [rbp-1F8h]
  int v27; // [rsp+14h] [rbp-1CCh] BYREF
  __int64 v28; // [rsp+18h] [rbp-1C8h] BYREF
  __int64 v29; // [rsp+20h] [rbp-1C0h] BYREF
  __int64 v30; // [rsp+28h] [rbp-1B8h] BYREF
  _BYTE v31[432]; // [rsp+30h] [rbp-1B0h] BYREF

  v7 = a3;
  v28 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v11 = sub_6E4240((__int64)a1, &v30);
  v12 = v11;
  if ( !v30 )
  {
    v12 = v11;
    v30 = sub_6E3DA0(v11, (__int64)v31);
  }
  if ( *(_BYTE *)(a2 + 56) )
    goto LABEL_4;
  if ( (unsigned int)sub_696370(v12) )
  {
    sub_6DC380(a1, a2, v7, a4, a5, 0);
    if ( !*(_BYTE *)(a2 + 56) )
    {
      v24 = v30;
      if ( (*((_BYTE *)a4 + 19) & 2) != 0 )
        goto LABEL_12;
      goto LABEL_15;
    }
  }
  else
  {
    v15 = *(_QWORD *)(a2 + 32);
    v16 = *(_QWORD *)(a2 + 24);
    v17 = *(_DWORD *)(a2 + 40);
    v26 = *(_QWORD *)(a2 + 48);
    v27 = 0;
    v18 = (__int64 *)sub_7410C0(v12, v16, v15, 0, (int)v30 + 68, v17, (__int64)&v27, v26, v28, (__int64)&v29);
    if ( v27 )
    {
      *(_BYTE *)(a2 + 56) = 1;
    }
    else if ( !*(_BYTE *)(a2 + 56) )
    {
      if ( v18 )
      {
        v19 = (__int64)a4;
        sub_6E7170(v18, (__int64)a4);
        if ( !(unsigned int)sub_8D32E0(*a4) )
          goto LABEL_11;
      }
      else
      {
        v25 = v29;
        v19 = (__int64)a4;
        if ( !v29 )
          v25 = v28;
        sub_6E6A50(v25, (__int64)a4);
        if ( !(unsigned int)sub_8D32E0(*a4) )
        {
LABEL_11:
          v24 = v30;
          if ( (*((_BYTE *)a4 + 19) & 2) != 0 )
          {
LABEL_12:
            *(_QWORD *)((char *)a4 + 68) = *(_QWORD *)(v24 + 68);
            *(_QWORD *)((char *)a4 + 76) = *(_QWORD *)(v24 + 76);
            return sub_724E30(&v28);
          }
LABEL_15:
          sub_6E4E90((__int64)a4, v24);
          return sub_724E30(&v28);
        }
      }
      sub_6F82C0((__int64)a4, v19, v20, v21, v22, v23);
      goto LABEL_11;
    }
  }
LABEL_4:
  sub_6E6260(a4);
  v13 = v30;
  *(_QWORD *)((char *)a4 + 68) = *(_QWORD *)(v30 + 68);
  *(_QWORD *)((char *)a4 + 76) = *(_QWORD *)(v13 + 76);
  return sub_724E30(&v28);
}
