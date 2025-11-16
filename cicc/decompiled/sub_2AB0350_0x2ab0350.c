// Function: sub_2AB0350
// Address: 0x2ab0350
//
void __fastcall sub_2AB0350(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rbx
  __int64 i; // r12
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // r14
  _BYTE *j; // rsi
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 v17; // r14
  __int64 v18; // rbx
  __int64 v19; // r12
  _QWORD *v20; // rax
  unsigned __int8 *v21; // r8
  __int64 v22; // rdx
  unsigned int v23; // r10d
  unsigned int v24; // edx
  bool v25; // al
  bool v26; // al
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rbx
  unsigned __int8 *v30; // rax
  unsigned __int8 *v31; // [rsp+8h] [rbp-D8h]
  __int64 v32; // [rsp+10h] [rbp-D0h]
  __int64 v33; // [rsp+10h] [rbp-D0h]
  bool v34; // [rsp+10h] [rbp-D0h]
  unsigned int v35; // [rsp+10h] [rbp-D0h]
  __int64 v36; // [rsp+18h] [rbp-C8h]
  __int64 v37; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v38; // [rsp+28h] [rbp-B8h] BYREF
  _QWORD v39[2]; // [rsp+30h] [rbp-B0h] BYREF
  const char *v40; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v41; // [rsp+48h] [rbp-98h]
  const char *v42; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v43; // [rsp+58h] [rbp-88h]
  char v44; // [rsp+70h] [rbp-70h]
  char v45; // [rsp+71h] [rbp-6Fh]
  __int64 v46; // [rsp+80h] [rbp-60h] BYREF
  char *v47; // [rsp+88h] [rbp-58h]
  __int64 v48; // [rsp+90h] [rbp-50h]
  int v49; // [rsp+98h] [rbp-48h]
  char v50; // [rsp+9Ch] [rbp-44h]
  char v51; // [rsp+A0h] [rbp-40h] BYREF

  v46 = 0;
  v47 = &v51;
  v48 = 2;
  v49 = 0;
  v50 = 1;
  v2 = sub_2BF3F10(a2);
  v3 = sub_2BF04D0(v2);
  v4 = sub_2BF05A0(v3);
  v7 = *(_QWORD *)(v3 + 120);
  for ( i = v4; v7 != i; v7 = *(_QWORD *)(v7 + 8) )
  {
    if ( !v7 )
      BUG();
    if ( *(_BYTE *)(v7 - 16) != 29 )
    {
      v9 = *(_QWORD *)(v7 - 8) & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)(v7 - 8) & 4) != 0 )
        v9 = **(_QWORD **)v9;
      sub_AE6EC0((__int64)&v46, *(_QWORD *)(v9 + 40));
    }
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(_QWORD *)(v10 + 120);
  v12 = v10 + 112;
  if ( v10 + 112 != v11 )
  {
    for ( j = *(_BYTE **)(v11 + 72); *j == 84; j = *(_BYTE **)(v14 + 72) )
    {
      v14 = *(_QWORD *)(v11 + 8);
      if ( !(unsigned __int8)sub_B19060((__int64)&v46, (__int64)j, v5, v6) )
      {
        v36 = sub_2BF0490(**(_QWORD **)(v11 + 24));
        sub_2C19E60(v11 - 24);
        sub_2C19E60(v36);
      }
      if ( v12 == v14 )
        break;
      v11 = v14;
    }
  }
  sub_2C37F10(a1);
  if ( LOBYTE(qword_500D260[17]) )
    sub_2C4B640(a1);
  v15 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(v15 + 64) != 1 )
    BUG();
  v16 = *(__int64 **)(v15 + 56);
  v17 = *v16;
  v18 = *(_QWORD *)(*v16 + 120);
  v19 = *v16 + 112;
  if ( v19 == v18 )
  {
LABEL_39:
    v45 = 1;
    v39[0] = v17;
    v39[1] = v18;
    v42 = "vec.epilog.resume.val";
    v44 = 3;
    v40 = (const char *)(a1 + 216);
    v27 = sub_2AAFF80(a1);
    v28 = 0;
    if ( *(_DWORD *)(v27 + 56) )
      v28 = **(_QWORD **)(v27 + 48);
    v41 = v28;
    v37 = 0;
    v38 = 0;
    v29 = sub_2AAFFE0(v39, 75, (__int64 *)&v40, 2, &v38, (void **)&v42);
    sub_9C6650(&v38);
    *(_QWORD *)(v29 + 136) = 0;
    sub_9C6650(&v37);
    goto LABEL_42;
  }
  while ( 1 )
  {
    if ( !v18 )
      BUG();
    if ( *(_BYTE *)(v18 - 16) != 4 )
      goto LABEL_19;
    if ( *(_BYTE *)(v18 + 136) != 75 )
      goto LABEL_19;
    v20 = *(_QWORD **)(v18 + 24);
    if ( a1 + 216 != *v20 )
      goto LABEL_19;
    v40 = 0;
    v41 = 64;
    v32 = v20[1];
    if ( !sub_2BF04A0(v32) )
    {
      v21 = *(unsigned __int8 **)(v32 + 40);
      if ( v21 )
      {
        v22 = *v21;
        if ( (_BYTE)v22 == 17 )
          break;
        if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v21 + 1) + 8LL) - 17 <= 1 && (unsigned __int8)v22 <= 0x15u )
        {
          v30 = sub_AD7630((__int64)v21, 0, v22);
          v21 = v30;
          if ( v30 )
          {
            if ( *v30 == 17 )
              break;
          }
        }
      }
    }
LABEL_25:
    if ( (unsigned int)v41 > 0x40 && v40 )
      j_j___libc_free_0_0((unsigned __int64)v40);
LABEL_19:
    v18 = *(_QWORD *)(v18 + 8);
    if ( v19 == v18 )
      goto LABEL_38;
  }
  v23 = *((_DWORD *)v21 + 8);
  v24 = v41;
  if ( v23 != (_DWORD)v41 )
  {
    v31 = v21;
    if ( v23 <= (unsigned int)v41 )
    {
      sub_C449B0((__int64)&v42, (const void **)v21 + 3, v41);
      if ( v43 <= 0x40 )
        v25 = v42 == v40;
      else
        v25 = sub_C43C50((__int64)&v42, (const void **)&v40);
    }
    else
    {
      v33 = (__int64)(v21 + 24);
      sub_C449B0((__int64)&v42, (const void **)&v40, v23);
      if ( *((_DWORD *)v31 + 8) <= 0x40u )
        v25 = *((_QWORD *)v31 + 3) == (_QWORD)v42;
      else
        v25 = sub_C43C50(v33, (const void **)&v42);
    }
    v34 = v25;
    sub_969240((__int64 *)&v42);
    v24 = v41;
    v26 = v34;
LABEL_35:
    if ( v26 )
      goto LABEL_36;
    goto LABEL_25;
  }
  if ( (unsigned int)v41 > 0x40 )
  {
    v35 = v41;
    v26 = sub_C43C50((__int64)(v21 + 24), (const void **)&v40);
    v24 = v35;
    goto LABEL_35;
  }
  if ( *((const char **)v21 + 3) != v40 )
    goto LABEL_25;
LABEL_36:
  if ( v24 > 0x40 && v40 )
    j_j___libc_free_0_0((unsigned __int64)v40);
  if ( v19 == v18 )
  {
LABEL_38:
    v18 = *(_QWORD *)(v17 + 120);
    goto LABEL_39;
  }
LABEL_42:
  if ( !v50 )
    _libc_free((unsigned __int64)v47);
}
