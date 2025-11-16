// Function: sub_324CF10
// Address: 0x324cf10
//
void __fastcall sub_324CF10(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int16 v5; // ax
  __int64 v6; // r14
  unsigned __int8 v7; // al
  __int64 *v8; // rdx
  __int64 v9; // rdx
  unsigned __int8 v10; // al
  __int64 *v11; // rdx
  const void *v12; // rcx
  size_t v13; // rdx
  size_t v14; // r8
  unsigned __int8 v15; // al
  __int64 v16; // rbx
  __int64 v17; // r15
  unsigned __int8 *v18; // rdx
  unsigned __int8 v19; // al
  unsigned __int8 v20; // al
  __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int64 **v23; // r12
  __int64 v24; // rax
  const void *v25; // rax
  size_t v26; // rdx
  __int64 v27; // [rsp+8h] [rbp-38h]

  v4 = a3 - 16;
  v5 = sub_AF18C0(a3);
  v6 = sub_324C6D0(a1, v5, a2, 0);
  if ( (unsigned __int16)sub_AF18C0(a3) == 48 )
  {
    v20 = *(_BYTE *)(a3 - 16);
    if ( (v20 & 2) != 0 )
      v21 = *(_QWORD *)(a3 - 32);
    else
      v21 = v4 - 8LL * ((v20 >> 2) & 0xF);
    sub_32495E0(a1, v6, *(_QWORD *)(v21 + 8), 73);
  }
  v7 = *(_BYTE *)(a3 - 16);
  if ( (v7 & 2) != 0 )
    v8 = *(__int64 **)(a3 - 32);
  else
    v8 = (__int64 *)(v4 - 8LL * ((v7 >> 2) & 0xF));
  if ( *v8 )
  {
    sub_B91420(*v8);
    if ( v9 )
    {
      v10 = *(_BYTE *)(a3 - 16);
      if ( (v10 & 2) != 0 )
        v11 = *(__int64 **)(a3 - 32);
      else
        v11 = (__int64 *)(v4 - 8LL * ((v10 >> 2) & 0xF));
      v12 = (const void *)*v11;
      if ( *v11 )
      {
        v12 = (const void *)sub_B91420(*v11);
        v14 = v13;
      }
      else
      {
        v14 = 0;
      }
      sub_324AD70(a1, v6, 3, v12, v14);
    }
  }
  if ( *(char *)(a3 + 1) < 0 && sub_3248C10((__int64)a1, 5u) )
    sub_3249FA0(a1, v6, 30);
  v15 = *(_BYTE *)(a3 - 16);
  if ( (v15 & 2) != 0 )
  {
    v16 = *(_QWORD *)(a3 - 32);
    v17 = *(_QWORD *)(v16 + 16);
    if ( !v17 )
      return;
  }
  else
  {
    v16 = v4 - 8LL * ((v15 >> 2) & 0xF);
    v17 = *(_QWORD *)(v16 + 16);
    if ( !v17 )
      return;
  }
  if ( *(_BYTE *)v17 != 1 )
    goto LABEL_27;
  v18 = *(unsigned __int8 **)(v17 + 136);
  v19 = *v18;
  if ( *v18 == 17 )
  {
    sub_324A3E0(a1, v6, (__int64)v18, *(char **)(v16 + 8));
    return;
  }
  if ( v19 == 18 )
  {
    sub_324A320(a1, v6, (__int64)v18);
    return;
  }
  if ( v19 <= 3u )
  {
    if ( (v18[33] & 3) != 1 )
    {
      v27 = *(_QWORD *)(v17 + 136);
      v22 = sub_A777F0(0x10u, a1 + 11);
      v23 = (unsigned __int64 **)v22;
      if ( v22 )
      {
        *(_QWORD *)v22 = 0;
        *(_DWORD *)(v22 + 8) = 0;
      }
      v24 = sub_31DB510(a1[23], v27);
      sub_324BB60(a1, v23, v24);
      sub_3249B00(a1, v23, 11, 159);
      sub_3249620(a1, v6, 2, (__int64 **)v23);
    }
  }
  else
  {
LABEL_27:
    if ( (unsigned __int16)sub_AF18C0(a3) == 16646 )
    {
      v25 = (const void *)sub_B91420(v17);
      sub_324AD70(a1, v6, 8464, v25, v26);
    }
    else if ( (unsigned __int16)sub_AF18C0(a3) == 16647 )
    {
      sub_324D230(a1, v6, v17);
    }
  }
}
