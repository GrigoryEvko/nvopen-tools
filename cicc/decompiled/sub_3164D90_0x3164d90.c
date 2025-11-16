// Function: sub_3164D90
// Address: 0x3164d90
//
void __fastcall sub_3164D90(__int64 a1, __int64 a2, char a3)
{
  __int64 v5; // r14
  __int64 v6; // rax
  char v7; // r15
  unsigned __int8 *v8; // r13
  _QWORD *v9; // rax
  unsigned __int8 *v10; // rbx
  unsigned __int8 *v11; // r15
  __int64 v12; // rsi
  unsigned __int8 *v13; // rsi
  __int64 v14; // rsi
  __int64 v15; // r13
  char v16; // r14
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // r8
  __int64 v20; // rax
  unsigned __int8 v21; // dl
  unsigned __int8 **v22; // rax
  unsigned __int8 *v23; // rbx
  __int64 v24; // rax
  unsigned __int8 v25; // dl
  unsigned __int8 **v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rbx
  __int64 v31; // rsi
  unsigned __int8 *v32; // rsi
  __int64 v33; // [rsp+8h] [rbp-98h]
  __int64 v34; // [rsp+18h] [rbp-88h] BYREF
  __int64 v35; // [rsp+20h] [rbp-80h] BYREF
  __int64 v36; // [rsp+28h] [rbp-78h] BYREF
  unsigned __int8 *v37[2]; // [rsp+30h] [rbp-70h] BYREF
  char v38; // [rsp+40h] [rbp-60h]
  unsigned __int8 *v39; // [rsp+50h] [rbp-50h] BYREF
  __int64 v40; // [rsp+58h] [rbp-48h]
  char v41; // [rsp+60h] [rbp-40h]

  v5 = a2 + 80;
  v6 = sub_B141A0(a2);
  v7 = *(_BYTE *)(a2 + 64) == 0;
  v33 = v6;
  v8 = (unsigned __int8 *)sub_B12A50(a2, 0);
  v9 = (_QWORD *)sub_B11F60(a2 + 80);
  sub_3164240((__int64)v37, a1, a3, v33, v8, v9, v7);
  if ( v38 )
  {
    v10 = v37[0];
    v11 = v37[1];
    sub_B13360(a2, v8, v37[0], 0);
    sub_B11F20(&v39, (__int64)v11);
    v12 = *(_QWORD *)(a2 + 80);
    if ( v12 )
      sub_B91220(v5, v12);
    v13 = v39;
    *(_QWORD *)(a2 + 80) = v39;
    if ( v13 )
      sub_B976B0((__int64)&v39, v13, v5);
    if ( !*(_BYTE *)(a2 + 64) )
    {
      if ( *v10 <= 0x1Cu )
      {
        if ( *v10 != 22 )
          return;
        v28 = *(_QWORD *)(v33 + 80);
        if ( !v28 )
          BUG();
        v15 = *(_QWORD *)(v28 + 32);
        LOWORD(v40) = 1;
        goto LABEL_25;
      }
      sub_B445D0((__int64)&v39, (char *)v10);
      v14 = *((_QWORD *)v10 + 6);
      v15 = (__int64)v39;
      v16 = v41;
      v34 = v14;
      if ( v14 )
        sub_B96E90((__int64)&v34, v14, 1);
      v17 = *(_QWORD *)(a2 + 24);
      v35 = v17;
      if ( v17 )
      {
        sub_B96E90((__int64)&v35, v17, 1);
        v18 = v34;
        v19 = v35;
        if ( !v34 )
        {
LABEL_20:
          if ( v19 )
          {
            sub_B91220((__int64)&v35, v19);
            v18 = v34;
            goto LABEL_22;
          }
          v29 = v34;
LABEL_31:
          v18 = v29;
LABEL_22:
          if ( !v18 )
            goto LABEL_24;
          goto LABEL_23;
        }
        if ( v35 )
        {
          v20 = sub_B10CD0((__int64)&v35);
          v21 = *(_BYTE *)(v20 - 16);
          if ( (v21 & 2) != 0 )
            v22 = *(unsigned __int8 ***)(v20 - 32);
          else
            v22 = (unsigned __int8 **)(v20 - 16 - 8LL * ((v21 >> 2) & 0xF));
          v23 = sub_AF34D0(*v22);
          v24 = sub_B10CD0((__int64)&v34);
          v25 = *(_BYTE *)(v24 - 16);
          if ( (v25 & 2) != 0 )
            v26 = *(unsigned __int8 ***)(v24 - 32);
          else
            v26 = (unsigned __int8 **)(v24 - 16 - 8LL * ((v25 >> 2) & 0xF));
          if ( v23 != sub_AF34D0(*v26) )
            goto LABEL_19;
          v36 = v34;
          if ( v34 )
          {
            v30 = a2 + 24;
            sub_B96E90((__int64)&v36, v34, 1);
            v31 = *(_QWORD *)(a2 + 24);
            if ( !v31 )
              goto LABEL_37;
          }
          else
          {
            v31 = *(_QWORD *)(a2 + 24);
            v30 = a2 + 24;
            if ( !v31 )
            {
LABEL_19:
              v19 = v35;
              goto LABEL_20;
            }
          }
          sub_B91220(v30, v31);
LABEL_37:
          v32 = (unsigned __int8 *)v36;
          *(_QWORD *)(a2 + 24) = v36;
          if ( v32 )
            sub_B976B0((__int64)&v36, v32, v30);
          goto LABEL_19;
        }
      }
      else
      {
        v29 = v34;
        v18 = v34;
        if ( !v34 )
          goto LABEL_31;
      }
LABEL_23:
      sub_B91220((__int64)&v34, v18);
LABEL_24:
      if ( !v16 )
        return;
LABEL_25:
      sub_B14260((_QWORD *)a2);
      if ( !v15 )
        BUG();
      v27 = *(_QWORD *)(v15 + 16);
      v39 = (unsigned __int8 *)v15;
      sub_AA8770(v27, a2, v15, v40);
    }
  }
}
