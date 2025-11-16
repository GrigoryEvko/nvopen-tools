// Function: sub_168D890
// Address: 0x168d890
//
_QWORD *__fastcall sub_168D890(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // rcx
  _QWORD *v4; // rdx
  _QWORD *v5; // rax
  _QWORD *v6; // r14
  _QWORD *v7; // r15
  _QWORD *v8; // rdi
  __int64 (__fastcall *v9)(__int64); // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  char *v12; // rsi
  _QWORD *v13; // r14
  _QWORD *v14; // r15
  _QWORD *v15; // rdi
  __int64 (__fastcall *v16)(__int64); // rax
  __int64 v17; // rdi
  _QWORD *v19; // [rsp+8h] [rbp-D8h]
  _QWORD *v20; // [rsp+10h] [rbp-D0h]
  _QWORD *v21; // [rsp+18h] [rbp-C8h]
  _QWORD *v23; // [rsp+28h] [rbp-B8h]
  _QWORD *v24; // [rsp+30h] [rbp-B0h] BYREF
  _QWORD *v25; // [rsp+38h] [rbp-A8h]
  _QWORD *v26; // [rsp+40h] [rbp-A0h]
  _QWORD *v27; // [rsp+48h] [rbp-98h]
  _QWORD *v28; // [rsp+50h] [rbp-90h]
  _QWORD *v29; // [rsp+58h] [rbp-88h]
  _QWORD *v30; // [rsp+60h] [rbp-80h]
  _QWORD *v31; // [rsp+68h] [rbp-78h]
  _QWORD v32[2]; // [rsp+70h] [rbp-70h] BYREF
  __int64 (__fastcall *v33)(__int64); // [rsp+80h] [rbp-60h]
  __int64 v34; // [rsp+88h] [rbp-58h]
  __int64 (__fastcall *v35)(__int64); // [rsp+90h] [rbp-50h]
  __int64 v36; // [rsp+98h] [rbp-48h]
  __int64 (__fastcall *v37)(__int64 *); // [rsp+A0h] [rbp-40h]
  __int64 v38; // [rsp+A8h] [rbp-38h]

  if ( !*(_QWORD *)a1 )
    *(_QWORD *)a1 = a2;
  v3 = (_QWORD *)a2[6];
  v4 = (_QWORD *)a2[2];
  v5 = (_QWORD *)a2[4];
  v20 = a2 + 7;
  v19 = a2 + 5;
  v21 = a2 + 1;
  v23 = a2 + 3;
  v24 = (_QWORD *)a2[8];
  v25 = a2 + 7;
  v26 = v3;
  v27 = a2 + 5;
  v28 = v4;
  v29 = a2 + 1;
  v30 = v5;
  v31 = a2 + 3;
  if ( a2 + 3 == v5 )
    goto LABEL_19;
  do
  {
    do
    {
      v6 = v32;
      v34 = 0;
      v36 = 0;
      v7 = v32;
      v8 = &v24;
      v33 = sub_12D3C60;
      v38 = 0;
      v35 = sub_12D3C80;
      v37 = sub_12D3CA0;
      v9 = sub_12D3C40;
      if ( ((unsigned __int8)sub_12D3C40 & 1) == 0 )
        goto LABEL_6;
      while ( 1 )
      {
        v9 = *(__int64 (__fastcall **)(__int64))((char *)v9 + *v8 - 1);
LABEL_6:
        v10 = v9((__int64)v8);
        if ( v10 )
          break;
        while ( 1 )
        {
          v11 = v7[3];
          v9 = (__int64 (__fastcall *)(__int64))v7[2];
          v6 += 2;
          v7 = v6;
          v8 = (_QWORD **)((char *)&v24 + v11);
          if ( ((unsigned __int8)v9 & 1) != 0 )
            break;
          v10 = v9((__int64)v8);
          if ( v10 )
            goto LABEL_9;
        }
      }
LABEL_9:
      v32[0] = v10;
      v12 = *(char **)(a1 + 120);
      if ( v12 == *(char **)(a1 + 128) )
      {
        sub_168D4A0((char **)(a1 + 112), v12, v32);
      }
      else
      {
        if ( v12 )
        {
          *(_QWORD *)v12 = v10;
          v12 = *(char **)(a1 + 120);
        }
        *(_QWORD *)(a1 + 120) = v12 + 8;
      }
      v13 = v32;
      v34 = 0;
      v36 = 0;
      v14 = v32;
      v15 = &v24;
      v33 = sub_12D3BB0;
      v38 = 0;
      v35 = sub_12D3BE0;
      v37 = sub_12D3C10;
      v16 = sub_12D3B80;
      if ( ((unsigned __int8)sub_12D3B80 & 1) != 0 )
        goto LABEL_14;
      while ( 2 )
      {
        if ( !(unsigned __int8)v16((__int64)v15) )
        {
          while ( 1 )
          {
            v17 = v14[3];
            v16 = (__int64 (__fastcall *)(__int64))v14[2];
            v13 += 2;
            v14 = v13;
            v15 = (_QWORD **)((char *)&v24 + v17);
            if ( ((unsigned __int8)v16 & 1) != 0 )
              break;
            if ( (unsigned __int8)v16((__int64)v15) )
              goto LABEL_18;
          }
LABEL_14:
          v16 = *(__int64 (__fastcall **)(__int64))((char *)v16 + *v15 - 1);
          continue;
        }
        break;
      }
LABEL_18:
      ;
    }
    while ( v23 != v30 );
LABEL_19:
    ;
  }
  while ( v23 != v31 || v21 != v28 || v21 != v29 || v19 != v26 || v19 != v27 || v20 != v24 || v20 != v25 );
  v32[0] = a1;
  return sub_168D470((__int64)a2, (__int64)sub_168D620, (__int64)v32);
}
