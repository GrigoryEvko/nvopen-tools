// Function: sub_2A6B9C0
// Address: 0x2a6b9c0
//
void __fastcall sub_2A6B9C0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // rbx
  unsigned __int8 *v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rcx
  int v8; // edx
  __int64 v9; // rcx
  int v10; // edx
  char v11; // al
  __int64 *v12; // r9
  unsigned __int8 *v13; // rcx
  _WORD *v14; // rsi
  _QWORD *v15; // rax
  unsigned __int8 v16; // bl
  unsigned int v17; // eax
  unsigned int v18; // esi
  unsigned int v19; // eax
  unsigned int v20; // r8d
  int v21; // eax
  _QWORD *v22; // rsi
  int v23; // eax
  int v24; // esi
  __int64 *v25; // rax
  unsigned __int8 *v26; // [rsp+20h] [rbp-100h]
  _BYTE *v27; // [rsp+20h] [rbp-100h]
  unsigned __int8 *v28; // [rsp+28h] [rbp-F8h]
  unsigned int v29; // [rsp+28h] [rbp-F8h]
  __int64 v30; // [rsp+30h] [rbp-F0h] BYREF
  unsigned int v31; // [rsp+38h] [rbp-E8h]
  __int64 v32; // [rsp+40h] [rbp-E0h] BYREF
  unsigned int v33; // [rsp+48h] [rbp-D8h]
  __int64 v34; // [rsp+50h] [rbp-D0h] BYREF
  unsigned int v35; // [rsp+58h] [rbp-C8h]
  __int64 v36; // [rsp+60h] [rbp-C0h] BYREF
  unsigned int v37; // [rsp+68h] [rbp-B8h]
  __int64 v38; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v39; // [rsp+78h] [rbp-A8h]
  __int64 v40; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v41; // [rsp+88h] [rbp-98h]
  unsigned __int8 v42[8]; // [rsp+90h] [rbp-90h] BYREF
  __int64 v43; // [rsp+98h] [rbp-88h] BYREF
  unsigned int v44; // [rsp+A0h] [rbp-80h]
  __int64 v45; // [rsp+A8h] [rbp-78h] BYREF
  unsigned int v46; // [rsp+B0h] [rbp-70h]
  __int64 v47[12]; // [rsp+C0h] [rbp-60h] BYREF

  v3 = a1 + 136;
  v47[0] = (__int64)a2;
  if ( *(_BYTE *)sub_2A686D0(a1 + 136, v47) != 6 )
  {
    v4 = (unsigned __int8 *)sub_2A68BC0(a1, *((unsigned __int8 **)a2 - 4));
    sub_22C05A0((__int64)v42, v4);
    if ( v42[0] <= 1u )
    {
LABEL_12:
      sub_22C0090(v42);
      return;
    }
    v5 = sub_2A637C0(a1, (__int64)v42, *(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL));
    if ( !v5 || (v6 = sub_96F480((unsigned int)*a2 - 29, v5, *((_QWORD *)a2 + 1), *(_QWORD *)a1)) == 0 )
    {
      v7 = *((_QWORD *)a2 + 1);
      v8 = *(unsigned __int8 *)(v7 + 8);
      if ( (unsigned int)(v8 - 17) <= 1 )
        LOBYTE(v8) = *(_BYTE *)(**(_QWORD **)(v7 + 16) + 8LL);
      if ( (_BYTE)v8 != 12 )
        goto LABEL_11;
      v9 = *(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL);
      v10 = *(unsigned __int8 *)(v9 + 8);
      if ( (unsigned int)(v10 - 17) <= 1 )
        LOBYTE(v10) = *(_BYTE *)(**(_QWORD **)(v9 + 16) + 8LL);
      if ( (_BYTE)v10 == 12 && *a2 != 78 )
      {
        v15 = sub_2A68BC0(a1, a2);
        v16 = v42[0];
        v27 = v15;
        if ( v42[0] == 4
          || (v17 = sub_BCB060(*(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL)), v18 = v17, v16 == 5)
          && (v29 = v17, v25 = sub_9876C0(&v43), v16 = v42[0], v18 = v29, v25) )
        {
          v31 = v44;
          if ( v44 > 0x40 )
            sub_C43780((__int64)&v30, (const void **)&v43);
          else
            v30 = v43;
          v33 = v46;
          if ( v46 > 0x40 )
            sub_C43780((__int64)&v32, (const void **)&v45);
          else
            v32 = v45;
        }
        else if ( v16 == 2 )
        {
          sub_AD8380((__int64)&v30, v43);
        }
        else if ( v16 )
        {
          sub_AADB10((__int64)&v30, v18, 1);
        }
        else
        {
          sub_AADB10((__int64)&v30, v18, 0);
        }
        v19 = sub_BCB060(*((_QWORD *)a2 + 1));
        sub_AB49F0((__int64)&v34, (__int64)&v30, *a2 - 29, v19);
        v39 = v35;
        if ( v35 > 0x40 )
          sub_C43780((__int64)&v38, (const void **)&v34);
        else
          v38 = v34;
        v41 = v37;
        if ( v37 > 0x40 )
          sub_C43780((__int64)&v40, (const void **)&v36);
        else
          v40 = v36;
        sub_22C06B0((__int64)v47, (__int64)&v38, 0);
        sub_2A639B0(a1, v27, (__int64)a2, (__int64)v47, 0x100000000LL);
        sub_22C0090((unsigned __int8 *)v47);
        sub_969240(&v40);
        sub_969240(&v38);
        sub_969240(&v36);
        sub_969240(&v34);
        sub_969240(&v32);
        sub_969240(&v30);
      }
      else
      {
LABEL_11:
        sub_2A6A450(a1, (__int64)a2);
      }
      goto LABEL_12;
    }
    v26 = (unsigned __int8 *)v6;
    v34 = (__int64)a2;
    v11 = sub_2A65730(v3, &v34, &v38);
    v12 = &v34;
    v13 = v26;
    if ( v11 )
    {
      v14 = (_WORD *)(v38 + 8);
LABEL_16:
      sub_2A63320(a1, (__int64)v14, (__int64)a2, v13, 0, (__int64)v12);
      sub_22C0090(v42);
      return;
    }
    v20 = *(_DWORD *)(a1 + 160);
    v21 = *(_DWORD *)(a1 + 152);
    v22 = (_QWORD *)v38;
    ++*(_QWORD *)(a1 + 136);
    v23 = v21 + 1;
    v47[0] = (__int64)v22;
    if ( 4 * v23 >= 3 * v20 )
    {
      v24 = 2 * v20;
      v28 = v26;
    }
    else
    {
      if ( v20 - *(_DWORD *)(a1 + 156) - v23 > v20 >> 3 )
      {
LABEL_30:
        *(_DWORD *)(a1 + 152) = v23;
        if ( *v22 != -4096 )
          --*(_DWORD *)(a1 + 156);
        v14 = v22 + 1;
        *((_QWORD *)v14 - 1) = v34;
        *v14 = 0;
        goto LABEL_16;
      }
      v24 = v20;
      v28 = v26;
    }
    sub_2A68410(v3, v24);
    sub_2A65730(v3, &v34, v47);
    v22 = (_QWORD *)v47[0];
    v13 = v28;
    v23 = *(_DWORD *)(a1 + 152) + 1;
    goto LABEL_30;
  }
}
