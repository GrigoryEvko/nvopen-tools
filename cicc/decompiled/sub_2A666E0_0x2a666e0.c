// Function: sub_2A666E0
// Address: 0x2a666e0
//
__int64 __fastcall sub_2A666E0(__int64 a1, __int64 a2)
{
  unsigned int v3; // ecx
  __int64 v4; // rdx
  unsigned int v5; // eax
  _QWORD *v6; // r14
  __int64 v7; // rsi
  _BYTE *v8; // r14
  _BYTE *v10; // rbx
  unsigned __int64 v11; // r12
  unsigned __int8 *v12; // rdi
  __int64 v13; // r13
  _BYTE *v14; // r15
  signed __int64 v15; // rax
  _BYTE *v16; // r14
  _BYTE *v17; // rbx
  __int64 v18; // r14
  __int64 v19; // rbx
  __int64 v20; // rbx
  __int64 v21; // r15
  bool v22; // zf
  __int64 v23; // rax
  __int64 v24; // rax
  _BYTE *v25; // rsi
  _BYTE *v26; // rdx
  unsigned __int8 v27; // al
  __int64 *v28; // rsi
  __int64 v29; // rdx
  unsigned __int8 *v30; // rbx
  unsigned __int8 *v31; // r12
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  int v34; // r8d
  _BYTE *v35; // [rsp+0h] [rbp-C0h]
  _BYTE *v36; // [rsp+0h] [rbp-C0h]
  _BYTE *v37; // [rsp+8h] [rbp-B8h]
  __int64 v38; // [rsp+18h] [rbp-A8h] BYREF
  _BYTE *v39; // [rsp+20h] [rbp-A0h] BYREF
  _BYTE *v40; // [rsp+28h] [rbp-98h]
  unsigned __int64 v41; // [rsp+40h] [rbp-80h] BYREF
  _BYTE *v42; // [rsp+48h] [rbp-78h]
  _BYTE *v43; // [rsp+50h] [rbp-70h]
  unsigned __int8 v44; // [rsp+60h] [rbp-60h] BYREF
  char v45; // [rsp+61h] [rbp-5Fh]
  unsigned __int64 v46; // [rsp+68h] [rbp-58h] BYREF
  unsigned int v47; // [rsp+70h] [rbp-50h]
  unsigned __int64 v48; // [rsp+78h] [rbp-48h] BYREF
  unsigned int v49; // [rsp+80h] [rbp-40h]

  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 15 )
  {
    sub_2A65CC0((unsigned __int64 *)&v39, a1, a2);
    v14 = v39;
    v37 = v40;
    v15 = 0xCCCCCCCCCCCCCCCDLL * ((v40 - v39) >> 3);
    if ( v15 >> 2 > 0 )
    {
      v16 = &v39[160 * (v15 >> 2)];
      do
      {
        if ( (unsigned __int8)sub_2A62E90(v14) )
          goto LABEL_21;
        v17 = v14;
        v14 += 40;
        if ( (unsigned __int8)sub_2A62E90(v14) )
          goto LABEL_21;
        v14 = v17 + 80;
        if ( (unsigned __int8)sub_2A62E90(v17 + 80) )
          goto LABEL_21;
        v14 = v17 + 120;
        if ( (unsigned __int8)sub_2A62E90(v17 + 120) )
          goto LABEL_21;
        v14 = v17 + 160;
      }
      while ( v16 != v17 + 160 );
      v15 = 0xCCCCCCCCCCCCCCCDLL * ((v37 - v14) >> 3);
    }
    if ( v15 != 2 )
    {
      if ( v15 != 3 )
      {
        if ( v15 != 1 )
          goto LABEL_22;
        goto LABEL_76;
      }
      if ( (unsigned __int8)sub_2A62E90(v14) )
        goto LABEL_21;
      v14 += 40;
    }
    if ( (unsigned __int8)sub_2A62E90(v14) )
      goto LABEL_21;
    v14 += 40;
LABEL_76:
    if ( (unsigned __int8)sub_2A62E90(v14) )
    {
LABEL_21:
      if ( v37 != v14 )
      {
        v10 = v40;
        v11 = (unsigned __int64)v39;
        if ( v40 != v39 )
        {
          do
          {
            v12 = (unsigned __int8 *)v11;
            v11 += 40LL;
            sub_22C0090(v12);
          }
          while ( v10 != (_BYTE *)v11 );
          v11 = (unsigned __int64)v39;
        }
        if ( v11 )
          j_j___libc_free_0(v11);
        return 0;
      }
    }
LABEL_22:
    v18 = *(_QWORD *)(a2 + 8);
    v42 = 0;
    v43 = 0;
    v19 = *(unsigned int *)(v18 + 12);
    v41 = 0;
    if ( !(_DWORD)v19 )
    {
      v29 = 0;
      v28 = 0;
LABEL_49:
      v13 = sub_AD24A0((__int64 **)v18, v28, v29);
      if ( v41 )
        j_j___libc_free_0(v41);
      v30 = v40;
      v31 = v39;
      if ( v40 != v39 )
      {
        do
        {
          while ( 1 )
          {
            if ( (unsigned int)*v31 - 4 <= 1 )
            {
              if ( *((_DWORD *)v31 + 8) > 0x40u )
              {
                v32 = *((_QWORD *)v31 + 3);
                if ( v32 )
                  j_j___libc_free_0_0(v32);
              }
              if ( *((_DWORD *)v31 + 4) > 0x40u )
              {
                v33 = *((_QWORD *)v31 + 1);
                if ( v33 )
                  break;
              }
            }
            v31 += 40;
            if ( v30 == v31 )
              goto LABEL_61;
          }
          j_j___libc_free_0_0(v33);
          v31 += 40;
        }
        while ( v30 != v31 );
LABEL_61:
        v31 = v39;
      }
      if ( v31 )
        j_j___libc_free_0((unsigned __int64)v31);
      return v13;
    }
    v20 = 8 * v19;
    v21 = 0;
    while ( 1 )
    {
      v26 = &v39[5 * v21];
      v27 = *v26;
      v45 = 0;
      v44 = v27;
      if ( v27 <= 3u )
        break;
      if ( (unsigned __int8)(v27 - 4) > 1u )
        goto LABEL_26;
      v47 = *((_DWORD *)v26 + 4);
      if ( v47 > 0x40 )
      {
        v36 = &v39[5 * v21];
        sub_C43780((__int64)&v46, (const void **)v26 + 1);
        v26 = v36;
      }
      else
      {
        v46 = *((_QWORD *)v26 + 1);
      }
      v49 = *((_DWORD *)v26 + 8);
      if ( v49 > 0x40 )
      {
        v35 = v26;
        sub_C43780((__int64)&v48, (const void **)v26 + 3);
        v26 = v35;
      }
      else
      {
        v48 = *((_QWORD *)v26 + 3);
      }
      v45 = v26[1];
      v22 = (unsigned __int8)sub_2A62D90((__int64)&v44) == 0;
      v23 = *(_QWORD *)(v18 + 16);
      if ( v22 )
      {
LABEL_41:
        v24 = sub_ACA8A0(*(__int64 ***)(v23 + v21));
        goto LABEL_28;
      }
LABEL_27:
      v24 = sub_2A637C0(a1, (__int64)&v44, *(_QWORD *)(v23 + v21));
LABEL_28:
      v38 = v24;
      v25 = v42;
      if ( v42 == v43 )
      {
        sub_262AD50((__int64)&v41, v42, &v38);
      }
      else
      {
        if ( v42 )
        {
          *(_QWORD *)v42 = v24;
          v25 = v42;
        }
        v42 = v25 + 8;
      }
      if ( (unsigned int)v44 - 4 > 1 )
        goto LABEL_33;
      if ( v49 > 0x40 && v48 )
        j_j___libc_free_0_0(v48);
      if ( v47 > 0x40 && v46 )
      {
        j_j___libc_free_0_0(v46);
        v21 += 8;
        if ( v20 == v21 )
        {
LABEL_48:
          v28 = (__int64 *)v41;
          v29 = (__int64)&v42[-v41] >> 3;
          goto LABEL_49;
        }
      }
      else
      {
LABEL_33:
        v21 += 8;
        if ( v20 == v21 )
          goto LABEL_48;
      }
    }
    if ( v27 > 1u )
      v46 = *((_QWORD *)v26 + 1);
LABEL_26:
    v22 = (unsigned __int8)sub_2A62D90((__int64)&v44) == 0;
    v23 = *(_QWORD *)(v18 + 16);
    if ( v22 )
      goto LABEL_41;
    goto LABEL_27;
  }
  v3 = *(_DWORD *)(a1 + 160);
  v4 = *(_QWORD *)(a1 + 144);
  if ( v3 )
  {
    v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (_QWORD *)(v4 + 48LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
    {
LABEL_4:
      v8 = v6 + 1;
      if ( !(unsigned __int8)sub_2A62E90(v8) )
        goto LABEL_5;
      return 0;
    }
    v34 = 1;
    while ( v7 != -4096 )
    {
      v5 = (v3 - 1) & (v34 + v5);
      v6 = (_QWORD *)(v4 + 48LL * v5);
      v7 = *v6;
      if ( a2 == *v6 )
        goto LABEL_4;
      ++v34;
    }
  }
  v8 = (_BYTE *)(v4 + 48LL * v3 + 8);
  if ( (unsigned __int8)sub_2A62E90(v8) )
    return 0;
LABEL_5:
  if ( (unsigned __int8)sub_2A62D90((__int64)v8) )
    return sub_2A637C0(a1, (__int64)v8, *(_QWORD *)(a2 + 8));
  else
    return sub_ACA8A0(*(__int64 ***)(a2 + 8));
}
