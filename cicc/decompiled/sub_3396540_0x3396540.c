// Function: sub_3396540
// Address: 0x3396540
//
void __fastcall sub_3396540(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // rax
  int v10; // edx
  int v11; // r8d
  int v12; // r9d
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // r11
  __int64 v16; // rsi
  __int64 v17; // r12
  int v18; // edx
  int v19; // r13d
  _QWORD *v20; // rax
  int v21; // edx
  bool v22; // al
  __int64 (*v23)(); // r9
  char v24; // al
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // r9
  __int64 v28; // rsi
  __int64 v29; // r12
  int v30; // edx
  int v31; // r13d
  int v32; // [rsp+0h] [rbp-90h]
  int v33; // [rsp+8h] [rbp-88h]
  int v34; // [rsp+8h] [rbp-88h]
  int v35; // [rsp+10h] [rbp-80h]
  int v36; // [rsp+10h] [rbp-80h]
  __int64 *v37; // [rsp+18h] [rbp-78h]
  unsigned int v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+48h] [rbp-48h] BYREF
  __int64 v40; // [rsp+50h] [rbp-40h] BYREF
  int v41; // [rsp+58h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v4 = sub_338B750(a1, *v3);
  v5 = *(_QWORD *)(a1 + 864);
  v7 = v6;
  v8 = *(_QWORD *)(v5 + 16);
  v37 = *(__int64 **)(a2 + 8);
  v9 = sub_2E79000(*(__int64 **)(v5 + 40));
  v38 = sub_2D5BAE0(v8, v9, v37, 0);
  v11 = v10;
  if ( *(_BYTE *)a2 <= 0x1Cu )
  {
    v12 = 0;
  }
  else
  {
    v12 = 0;
    if ( ((*(_BYTE *)a2 - 68) & 0xFB) == 0 )
    {
      v34 = v10;
      v22 = sub_B44910(a2);
      v11 = v34;
      v12 = 0;
      if ( v22 )
      {
        v23 = *(__int64 (**)())(*(_QWORD *)v8 + 1456LL);
        if ( v23 != sub_2D56680 )
        {
          v24 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD))v23)(
                  v8,
                  *(unsigned __int16 *)(*(_QWORD *)(v4 + 48) + 16LL * (unsigned int)v7),
                  *(_QWORD *)(*(_QWORD *)(v4 + 48) + 16LL * (unsigned int)v7 + 8),
                  v38);
          v11 = v34;
          if ( v24 )
          {
            v25 = *(_DWORD *)(a1 + 848);
            v26 = *(_QWORD *)a1;
            v40 = 0;
            v27 = *(_QWORD *)(a1 + 864);
            v41 = v25;
            if ( v26 )
            {
              if ( &v40 != (__int64 *)(v26 + 48) )
              {
                v28 = *(_QWORD *)(v26 + 48);
                v40 = v28;
                if ( v28 )
                {
                  v36 = v27;
                  sub_B96E90((__int64)&v40, v28, 1);
                  v11 = v34;
                  LODWORD(v27) = v36;
                }
              }
            }
            v39 = a2;
            v29 = sub_33FAF80(v27, 213, (unsigned int)&v40, v38, v11, v27);
            v31 = v30;
            v20 = sub_337DC20(a1 + 8, &v39);
            *v20 = v29;
            v21 = v31;
            goto LABEL_10;
          }
        }
        v12 = 16;
      }
    }
  }
  v13 = *(_DWORD *)(a1 + 848);
  v14 = *(_QWORD *)a1;
  v40 = 0;
  v15 = *(_QWORD *)(a1 + 864);
  v41 = v13;
  if ( v14 )
  {
    if ( &v40 != (__int64 *)(v14 + 48) )
    {
      v16 = *(_QWORD *)(v14 + 48);
      v40 = v16;
      if ( v16 )
      {
        v35 = v15;
        v32 = v11;
        v33 = v12;
        sub_B96E90((__int64)&v40, v16, 1);
        v11 = v32;
        v12 = v33;
        LODWORD(v15) = v35;
      }
    }
  }
  v39 = a2;
  v17 = sub_33FA050(v15, 214, (unsigned int)&v40, v38, v11, v12, v4, v7);
  v19 = v18;
  v20 = sub_337DC20(a1 + 8, &v39);
  *v20 = v17;
  v21 = v19;
LABEL_10:
  *((_DWORD *)v20 + 2) = v21;
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
}
