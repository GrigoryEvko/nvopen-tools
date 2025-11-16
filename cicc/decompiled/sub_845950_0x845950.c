// Function: sub_845950
// Address: 0x845950
//
void __fastcall sub_845950(_QWORD *a1, __int64 a2, int a3, int a4, __int64 a5)
{
  int v8; // r15d
  int v9; // eax
  __int64 v10; // r8
  char v11; // dl
  __m128i *v12; // rsi
  _QWORD *v13; // rax
  unsigned int v14; // eax
  const __m128i *i; // r13
  __m128i *v16; // rbx
  char v17; // al
  __int64 v18; // rax
  __m128i *v19; // rax
  _QWORD *v20; // r13
  __int64 v21; // [rsp+0h] [rbp-80h]
  char v22; // [rsp+Ch] [rbp-74h]
  bool v23; // [rsp+Ch] [rbp-74h]
  int v24; // [rsp+14h] [rbp-6Ch] BYREF
  const __m128i *v25; // [rsp+18h] [rbp-68h] BYREF
  _BYTE v26[96]; // [rsp+20h] [rbp-60h] BYREF

  v21 = *(_QWORD *)(a2 + 48);
  v8 = (*(_BYTE *)(a5 + 18) != 0) << 8;
  v22 = *(_BYTE *)(v21 + a3 - 1);
  v9 = sub_8E31E0(*a1);
  v11 = v22;
  if ( !v9 && v22 != 67 )
    return;
  v23 = 0;
  if ( a4 )
  {
    v23 = (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x10) != 0;
    *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x10u;
  }
  v12 = *(__m128i **)(a2 + 112);
  if ( v12 )
  {
    if ( v11 == 79 )
    {
      if ( *(_BYTE *)(v21 + 1) == 77 )
      {
        v13 = (_QWORD *)sub_8D4890(*(_QWORD *)(a2 + 112));
        v12 = (__m128i *)sub_72D2E0(v13);
      }
      goto LABEL_9;
    }
    if ( v11 != 67 || *(_QWORD *)(a5 + 48) )
    {
LABEL_9:
      sub_8453D0((__m128i *)a1, v12, 0, (_BYTE *)(a5 + 48), 1u, 1u, v8, 0, (FILE *)((char *)a1 + 68));
      goto LABEL_10;
    }
    v17 = *(_BYTE *)(a5 + 64);
    if ( (v17 & 0x10) != 0 )
    {
      sub_831640((__m128i *)a1, v12, *(_QWORD *)(a5 + 72), v21, v10);
      v17 = *(_BYTE *)(a5 + 64);
    }
    if ( (v17 & 4) == 0 )
      sub_6FA3A0((__m128i *)a1, (__int64)v12);
LABEL_10:
    if ( !a4 )
      return;
    goto LABEL_11;
  }
  if ( v11 == 66 )
  {
    v18 = sub_72C390();
    v11 = 66;
    v12 = (__m128i *)v18;
    goto LABEL_36;
  }
  if ( v11 == 68 )
  {
    v19 = (__m128i *)sub_72BA30(unk_4F06A60);
    v11 = 68;
    v12 = v19;
LABEL_36:
    if ( v12 )
      goto LABEL_9;
  }
  if ( (*(_BYTE *)(a5 + 64) & 0x88) != 0 )
  {
    v14 = sub_8274A0(v11);
    if ( (unsigned int)sub_840360(a1, *(_QWORD *)(a5 + 40), v14, 0, 1, 1, 0, 0, v8, (__int64)v26, &v24, &v25) )
      sub_721090();
    if ( v25 )
    {
      if ( (unsigned int)sub_6E5430() )
      {
        v20 = sub_67DA80(0x1A2u, (_DWORD *)a1 + 17, *a1);
        sub_82E650(v25->m128i_i64, 0, 0, 0, v20);
        sub_685910((__int64)v20, 0);
      }
      for ( i = v25; i; qword_4D03C68 = v16->m128i_i64 )
      {
        v16 = (__m128i *)i;
        i = (const __m128i *)i->m128i_i64[0];
        sub_725130((__int64 *)v16[2].m128i_i64[1]);
        sub_82D8A0((_QWORD *)v16[7].m128i_i64[1]);
        v16->m128i_i64[0] = (__int64)qword_4D03C68;
      }
    }
    sub_6E6840((__int64)a1);
    goto LABEL_10;
  }
  if ( !*(_QWORD *)(a5 + 48) )
    sub_6F69D0(a1, 8u);
  sub_8449E0(a1, 0, a5 + 48, 0, 0);
  if ( a4 )
  {
    if ( (unsigned int)sub_8D2D80(*a1) )
      sub_6FE880((__m128i *)a1, 0);
LABEL_11:
    *(_BYTE *)(qword_4D03C50 + 18LL) = *(_BYTE *)(qword_4D03C50 + 18LL) & 0xEF | (16 * v23);
  }
}
