// Function: sub_28CBF00
// Address: 0x28cbf00
//
__int64 __fastcall sub_28CBF00(__int64 a1, __int64 a2)
{
  unsigned __int8 *v4; // rsi
  __int64 v5; // rax
  unsigned __int8 **v6; // rdx
  __int64 v7; // rax
  bool v8; // zf
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int8 **v11; // rax
  unsigned __int8 *v12; // rcx
  unsigned __int8 *v13; // r13
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int8 **v18; // rdx
  __int64 v19; // rax
  unsigned __int8 **v20; // r14
  unsigned __int8 **v21; // r13
  __int64 v22; // rbx
  __int64 *v23; // rsi
  unsigned int v24; // eax
  int v25; // r9d
  __int64 v26; // r11
  unsigned int v27; // r8d
  int v28; // r10d
  unsigned int v29; // ebx
  __int64 v30; // rdi
  unsigned __int8 *v31; // r15
  unsigned int v32; // edi
  int v33; // edi
  int v34; // r14d
  __int64 v35; // rax
  __int64 v36; // [rsp+8h] [rbp-98h]
  unsigned __int8 **v37; // [rsp+10h] [rbp-90h]
  unsigned int v38; // [rsp+1Ch] [rbp-84h]
  unsigned __int8 *v39; // [rsp+28h] [rbp-78h] BYREF
  unsigned __int8 **v40; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int8 **v41; // [rsp+38h] [rbp-68h]
  __int64 v42; // [rsp+40h] [rbp-60h]
  __int64 v43; // [rsp+48h] [rbp-58h]
  __int64 v44; // [rsp+50h] [rbp-50h] BYREF
  __int64 v45; // [rsp+58h] [rbp-48h]
  __int64 v46; // [rsp+60h] [rbp-40h]
  __int64 v47; // [rsp+68h] [rbp-38h]

  if ( *(_DWORD *)(a2 + 176) )
  {
    v4 = *(unsigned __int8 **)(a2 + 24);
    if ( v4 && *v4 == 62 )
      return sub_28C8480(a1, (__int64)v4);
    v5 = *(_QWORD *)(a2 + 72);
    if ( *(_BYTE *)(a2 + 92) )
      v6 = (unsigned __int8 **)(v5 + 8LL * *(unsigned int *)(a2 + 84));
    else
      v6 = (unsigned __int8 **)(v5 + 8LL * *(unsigned int *)(a2 + 80));
    v40 = *(unsigned __int8 ***)(a2 + 72);
    v41 = v6;
    sub_254BBF0((__int64)&v40);
    v7 = *(_QWORD *)(a2 + 64);
    v8 = *(_BYTE *)(a2 + 92) == 0;
    v42 = a2 + 64;
    v43 = v7;
    if ( v8 )
      v9 = *(unsigned int *)(a2 + 80);
    else
      v9 = *(unsigned int *)(a2 + 84);
    v44 = *(_QWORD *)(a2 + 72) + 8 * v9;
    v45 = v44;
    sub_254BBF0((__int64)&v44);
    v10 = *(_QWORD *)(a2 + 64);
    v46 = a2 + 64;
    v47 = v10;
    v11 = v40;
    if ( (unsigned __int8 **)v44 == v40 )
      goto LABEL_14;
    while ( 1 )
    {
      v12 = *v11;
      if ( **v11 == 62 )
        break;
      do
        ++v11;
      while ( v41 != v11 && (unsigned __int64)*v11 >= 0xFFFFFFFFFFFFFFFELL );
      if ( (unsigned __int8 **)v44 == v11 )
        goto LABEL_14;
    }
    if ( (unsigned __int8 **)v44 == v11 )
    {
LABEL_14:
      v13 = 0;
LABEL_15:
      v4 = v13;
      return sub_28C8480(a1, (__int64)v4);
    }
    v25 = *(_DWORD *)(a1 + 2440);
    v13 = 0;
    v26 = *(_QWORD *)(a1 + 2424);
    v27 = -1;
    v28 = v25 - 1;
    if ( v25 )
    {
LABEL_37:
      v29 = v28 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v30 = v26 + 16LL * v29;
      v31 = *(unsigned __int8 **)v30;
      if ( v12 == *(unsigned __int8 **)v30 )
      {
LABEL_38:
        v32 = *(_DWORD *)(v30 + 8);
        goto LABEL_39;
      }
      v33 = 1;
      while ( v31 != (unsigned __int8 *)-4096LL )
      {
        v34 = v33 + 1;
        v29 = v28 & (v33 + v29);
        v30 = v26 + 16LL * v29;
        v31 = *(unsigned __int8 **)v30;
        if ( *(unsigned __int8 **)v30 == v12 )
          goto LABEL_38;
        v33 = v34;
      }
    }
LABEL_50:
    v32 = 0;
LABEL_39:
    if ( v27 > v32 )
    {
      v27 = v32;
      v13 = v12;
    }
    do
      ++v11;
    while ( v41 != v11 && (unsigned __int64)(*v11 + 2) <= 1 );
    while ( (unsigned __int8 **)v44 != v11 )
    {
      v12 = *v11;
      if ( **v11 == 62 )
      {
        if ( (unsigned __int8 **)v44 == v11 )
          goto LABEL_15;
        if ( v25 )
          goto LABEL_37;
        goto LABEL_50;
      }
      do
        ++v11;
      while ( v41 != v11 && (unsigned __int64)*v11 >= 0xFFFFFFFFFFFFFFFELL );
    }
    goto LABEL_15;
  }
  v15 = *(unsigned int *)(a2 + 148);
  if ( *(_DWORD *)(a2 + 148) - *(_DWORD *)(a2 + 152) == 1 )
  {
    v44 = *(_QWORD *)(a2 + 136);
    v45 = sub_254BB00(a2 + 128);
    sub_254BBF0((__int64)&v44);
    v35 = *(_QWORD *)(a2 + 128);
    v46 = a2 + 128;
    v47 = v35;
    return *(_QWORD *)v44;
  }
  else
  {
    if ( !*(_BYTE *)(a2 + 156) )
      v15 = *(unsigned int *)(a2 + 144);
    v44 = *(_QWORD *)(a2 + 136) + 8 * v15;
    v45 = v44;
    sub_254BBF0((__int64)&v44);
    v16 = *(_QWORD *)(a2 + 128);
    v8 = *(_BYTE *)(a2 + 156) == 0;
    v46 = a2 + 128;
    v47 = v16;
    v17 = *(_QWORD *)(a2 + 136);
    if ( v8 )
      v18 = (unsigned __int8 **)(v17 + 8LL * *(unsigned int *)(a2 + 144));
    else
      v18 = (unsigned __int8 **)(v17 + 8LL * *(unsigned int *)(a2 + 148));
    v40 = *(unsigned __int8 ***)(a2 + 136);
    v41 = v18;
    sub_254BBF0((__int64)&v40);
    v19 = *(_QWORD *)(a2 + 128);
    v20 = v40;
    v42 = a2 + 128;
    v36 = 0;
    v21 = v41;
    v43 = v19;
    v37 = (unsigned __int8 **)v44;
    if ( v40 != (unsigned __int8 **)v44 )
    {
      v38 = -1;
      do
      {
        v22 = (__int64)*v20;
        if ( (unsigned int)**v20 - 26 > 1 )
          v39 = *v20;
        else
          v39 = *(unsigned __int8 **)(v22 + 72);
        v23 = sub_28CBE90(a1 + 2416, (__int64 *)&v39);
        v24 = 0;
        if ( v23 )
          v24 = *((_DWORD *)v23 + 2);
        if ( v38 > v24 )
        {
          v38 = v24;
          v36 = v22;
        }
        for ( ++v20; v21 != v20; ++v20 )
        {
          if ( (unsigned __int64)*v20 < 0xFFFFFFFFFFFFFFFELL )
            break;
        }
      }
      while ( v37 != v20 );
    }
  }
  return v36;
}
