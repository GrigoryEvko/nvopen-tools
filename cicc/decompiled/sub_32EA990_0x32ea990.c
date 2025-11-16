// Function: sub_32EA990
// Address: 0x32ea990
//
__int64 __fastcall sub_32EA990(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, _BYTE *a6)
{
  __int64 v10; // rsi
  int v11; // eax
  unsigned __int8 v12; // si
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdi
  int v16; // esi
  __int64 v17; // r13
  __int64 v19; // rax
  unsigned __int16 v20; // dx
  __int64 v21; // rax
  __int64 v22; // rax
  int v23; // esi
  __int64 v24; // rax
  __int128 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int128 v29; // rax
  __int64 v30; // rax
  __int128 v31; // [rsp-10h] [rbp-90h]
  __int128 v32; // [rsp-10h] [rbp-90h]
  _BYTE *v33; // [rsp+0h] [rbp-80h]
  __int64 v34; // [rsp+8h] [rbp-78h]
  int v35; // [rsp+8h] [rbp-78h]
  int v36; // [rsp+8h] [rbp-78h]
  int v37; // [rsp+8h] [rbp-78h]
  __int64 v38; // [rsp+10h] [rbp-70h] BYREF
  int v39; // [rsp+18h] [rbp-68h]
  unsigned __int16 v40; // [rsp+20h] [rbp-60h] BYREF
  __int64 v41; // [rsp+28h] [rbp-58h]

  *a6 = 0;
  v10 = *(_QWORD *)(a2 + 80);
  v38 = v10;
  if ( v10 )
  {
    v33 = a6;
    v34 = a5;
    sub_B96E90((__int64)&v38, v10, 1);
    a6 = v33;
    a5 = v34;
  }
  v39 = *(_DWORD *)(a2 + 72);
  v11 = *(_DWORD *)(a2 + 24);
  if ( v11 != 298 )
  {
    if ( v11 == 4 )
    {
      v37 = a5;
      *(_QWORD *)&v29 = sub_32EA820(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), a4, a5);
      LODWORD(a5) = v37;
      if ( !(_QWORD)v29 )
        goto LABEL_22;
      v26 = sub_3406EB0(
              *a1,
              4,
              (unsigned int)&v38,
              a4,
              v37,
              (unsigned int)&v38,
              v29,
              *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
    }
    else
    {
      if ( v11 == 11 )
      {
        v19 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
        v20 = *(_WORD *)v19;
        v21 = *(_QWORD *)(v19 + 8);
        v40 = v20;
        v41 = v21;
        if ( v20 )
        {
          if ( v20 == 1 || (unsigned __int16)(v20 - 504) <= 7u )
            BUG();
          v30 = *(_QWORD *)&byte_444C4A0[16 * v20 - 16];
          if ( !v30 )
            goto LABEL_15;
        }
        else
        {
          v35 = a5;
          v22 = sub_3007260((__int64)&v40);
          LODWORD(a5) = v35;
          if ( !v22 )
          {
LABEL_15:
            v23 = 214;
LABEL_16:
            *((_QWORD *)&v31 + 1) = a3;
            *(_QWORD *)&v31 = a2;
            v24 = sub_33FAF80(*a1, v23, (unsigned int)&v38, a4, a5, (_DWORD)a6, v31);
            goto LABEL_17;
          }
          LOBYTE(v30) = sub_3007260((__int64)&v40);
          LODWORD(a5) = v35;
        }
        v23 = ((v30 & 7) != 0) + 213;
        goto LABEL_16;
      }
      if ( v11 != 3
        || (v36 = a5,
            *(_QWORD *)&v25 = sub_32EAC70(
                                a1,
                                **(_QWORD **)(a2 + 40),
                                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                                a4,
                                a5,
                                a6),
            LODWORD(a5) = v36,
            !(_QWORD)v25) )
      {
LABEL_22:
        v27 = a1[1];
        v28 = 1;
        if ( (_WORD)a4 != 1
          && (!(_WORD)a4 || (v28 = (unsigned __int16)a4, !*(_QWORD *)(v27 + 8LL * (unsigned __int16)a4 + 112)))
          || *(_BYTE *)(v27 + 500 * v28 + 6629) )
        {
          v17 = 0;
          goto LABEL_8;
        }
        *((_QWORD *)&v32 + 1) = a3;
        *(_QWORD *)&v32 = a2;
        v24 = sub_33FAF80(*a1, 215, (unsigned int)&v38, a4, a5, (_DWORD)a6, v32);
LABEL_17:
        v17 = v24;
        goto LABEL_8;
      }
      v26 = sub_3406EB0(
              *a1,
              3,
              (unsigned int)&v38,
              a4,
              v36,
              (unsigned int)&v38,
              v25,
              *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
    }
    v17 = v26;
    goto LABEL_8;
  }
  if ( (*(_WORD *)(a2 + 32) & 0x380) != 0 )
    goto LABEL_22;
  v12 = *(_BYTE *)(a2 + 33);
  v13 = *(_QWORD *)(a2 + 104);
  v14 = *(unsigned __int16 *)(a2 + 96);
  *a6 = 1;
  v15 = *a1;
  v16 = (v12 >> 2) & 3;
  if ( !v16 )
    LOBYTE(v16) = 1;
  v17 = sub_33F1B30(
          v15,
          (unsigned __int8)v16,
          (unsigned int)&v38,
          a4,
          a5,
          *(_QWORD *)(a2 + 112),
          **(_QWORD **)(a2 + 40),
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
          v14,
          v13);
LABEL_8:
  if ( v38 )
    sub_B91220((__int64)&v38, v38);
  return v17;
}
