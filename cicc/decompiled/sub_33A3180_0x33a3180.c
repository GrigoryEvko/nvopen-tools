// Function: sub_33A3180
// Address: 0x33a3180
//
void __fastcall sub_33A3180(__int64 a1, __int64 a2, int a3)
{
  int v5; // edx
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rbx
  int v10; // edx
  int v11; // r13d
  _QWORD *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // rbx
  __int64 v19; // rax
  unsigned int *v20; // rax
  __int64 v21; // rbx
  int v22; // r9d
  __int64 v23; // rbx
  int v24; // edx
  int v25; // r13d
  _QWORD *v26; // rax
  __int64 v27; // rax
  __int64 v28; // rbx
  int v29; // edx
  int v30; // r13d
  _QWORD *v31; // rax
  __int64 *v32; // [rsp+8h] [rbp-88h]
  __int64 v33; // [rsp+48h] [rbp-48h] BYREF
  __int64 v34; // [rsp+50h] [rbp-40h] BYREF
  int v35; // [rsp+58h] [rbp-38h]

  v5 = *(_DWORD *)(a1 + 848);
  v6 = *(_QWORD *)a1;
  v34 = 0;
  v35 = v5;
  if ( v6 && &v34 != (__int64 *)(v6 + 48) && (v7 = *(_QWORD *)(v6 + 48), (v34 = v7) != 0) )
  {
    sub_B96E90((__int64)&v34, v7, 1);
    if ( a3 != 143 )
    {
      if ( a3 != 144 )
      {
        if ( a3 != 142 )
          goto LABEL_8;
        goto LABEL_7;
      }
      goto LABEL_15;
    }
  }
  else if ( a3 != 143 )
  {
    if ( a3 != 144 )
    {
      if ( a3 != 142 )
        return;
LABEL_7:
      v8 = sub_33F17F0(*(_QWORD *)(a1 + 864), 493, &v34, 264, 0);
      v33 = a2;
      v9 = v8;
      v11 = v10;
      v12 = sub_337DC20(a1 + 8, &v33);
      *v12 = v9;
      *((_DWORD *)v12 + 2) = v11;
      goto LABEL_8;
    }
LABEL_15:
    if ( *(char *)(a2 + 7) < 0 )
    {
      v13 = sub_BD2BC0(a2);
      v15 = v13 + v14;
      if ( *(char *)(a2 + 7) < 0 )
        v15 -= sub_BD2BC0(a2);
      v16 = v15 >> 4;
      if ( (_DWORD)v16 )
      {
        v17 = 0;
        v18 = 16LL * (unsigned int)v16;
        while ( 1 )
        {
          v19 = 0;
          if ( *(char *)(a2 + 7) < 0 )
            v19 = sub_BD2BC0(a2);
          v20 = (unsigned int *)(v17 + v19);
          if ( *(_DWORD *)(*(_QWORD *)v20 + 8LL) == 9 )
            break;
          v17 += 16;
          if ( v18 == v17 )
            goto LABEL_25;
        }
        v32 = (__int64 *)(a2 + 32 * (v20[2] - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
      }
    }
LABEL_25:
    v21 = *(_QWORD *)(a1 + 864);
    sub_338B750(a1, *v32);
    v33 = a2;
    v23 = sub_33FAF80(v21, 495, (unsigned int)&v34, 264, 0, v22);
    v25 = v24;
    v26 = sub_337DC20(a1 + 8, &v33);
    *v26 = v23;
    *((_DWORD *)v26 + 2) = v25;
    goto LABEL_8;
  }
  v27 = sub_33F17F0(*(_QWORD *)(a1 + 864), 494, &v34, 264, 0);
  v33 = a2;
  v28 = v27;
  v30 = v29;
  v31 = sub_337DC20(a1 + 8, &v33);
  *v31 = v28;
  *((_DWORD *)v31 + 2) = v30;
LABEL_8:
  if ( v34 )
    sub_B91220((__int64)&v34, v34);
}
