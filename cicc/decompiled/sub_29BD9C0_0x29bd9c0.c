// Function: sub_29BD9C0
// Address: 0x29bd9c0
//
__int64 __fastcall sub_29BD9C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rbx
  __int64 *v7; // rbx
  unsigned int v8; // esi
  __int64 v9; // rax
  unsigned int v10; // ecx
  __int64 *v11; // rax
  __int64 *v12; // rdx
  unsigned int v13; // r12d
  _QWORD *v14; // rsi
  __int64 v15; // rcx
  unsigned int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // r15
  char *v19; // rax
  unsigned int v20; // eax
  __int64 v21; // r15
  __int64 v22; // rdx
  __int64 *v23; // r14
  char *v24; // rax
  char *v25; // rdx
  unsigned __int64 v26; // rdi
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  _QWORD *v32; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v33; // [rsp+28h] [rbp-D8h]
  _QWORD v34[8]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v35; // [rsp+70h] [rbp-90h] BYREF
  char *v36; // [rsp+78h] [rbp-88h]
  __int64 v37; // [rsp+80h] [rbp-80h]
  int v38; // [rsp+88h] [rbp-78h]
  unsigned __int8 v39; // [rsp+8Ch] [rbp-74h]
  char v40; // [rsp+90h] [rbp-70h] BYREF

  v6 = *(__int64 **)(*(_QWORD *)(a1 + 72) + 80LL);
  if ( !v6 )
  {
    v13 = 0;
    if ( !a2 )
      return v13;
    v8 = *(_DWORD *)(a3 + 32);
    v9 = (unsigned int)(*(_DWORD *)(a1 + 44) + 1);
    if ( v8 <= (unsigned int)v9 )
      goto LABEL_7;
LABEL_5:
    v6 = *(__int64 **)(*(_QWORD *)(a3 + 24) + 8 * v9);
LABEL_6:
    if ( !a2 )
    {
      a5 = 0;
      v10 = 0;
LABEL_8:
      v11 = 0;
      if ( v10 < v8 )
        v11 = *(__int64 **)(*(_QWORD *)(a3 + 24) + 8 * a5);
      while ( v6 != v11 )
      {
        if ( *((_DWORD *)v6 + 4) < *((_DWORD *)v11 + 4) )
        {
          v12 = v6;
          v6 = v11;
          v11 = v12;
        }
        v6 = (__int64 *)v6[1];
      }
      v7 = (__int64 *)*v11;
      v13 = 0;
      if ( !*v11 )
        return v13;
      goto LABEL_15;
    }
LABEL_7:
    a5 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    v10 = *(_DWORD *)(a2 + 44) + 1;
    goto LABEL_8;
  }
  v7 = v6 - 3;
  if ( (__int64 *)a1 != v7 && (__int64 *)a2 != v7 )
  {
    v8 = *(_DWORD *)(a3 + 32);
    v6 = 0;
    v9 = (unsigned int)(*(_DWORD *)(a1 + 44) + 1);
    if ( (unsigned int)v9 >= v8 )
      goto LABEL_6;
    goto LABEL_5;
  }
LABEL_15:
  v14 = v34;
  v34[0] = a1;
  v15 = 1;
  v39 = 1;
  v32 = v34;
  v35 = 0;
  v37 = 8;
  v38 = 0;
  v36 = &v40;
  v33 = 0x800000001LL;
  v16 = 1;
  while ( 1 )
  {
    v17 = v16;
    v18 = v14[v16 - 1];
    LODWORD(v33) = v16 - 1;
    if ( !(_BYTE)v15 )
    {
LABEL_46:
      sub_C8CC70((__int64)&v35, v18, v17, v15, a5, a6);
      goto LABEL_21;
    }
    v19 = v36;
    v15 = HIDWORD(v37);
    v17 = (__int64)&v36[8 * HIDWORD(v37)];
    if ( v36 == (char *)v17 )
    {
LABEL_47:
      if ( HIDWORD(v37) >= (unsigned int)v37 )
        goto LABEL_46;
      ++HIDWORD(v37);
      *(_QWORD *)v17 = v18;
      ++v35;
    }
    else
    {
      while ( v18 != *(_QWORD *)v19 )
      {
        v19 += 8;
        if ( (char *)v17 == v19 )
          goto LABEL_47;
      }
    }
LABEL_21:
    sub_B19AA0(a4, v18, a2);
    v13 = v20;
    if ( (_BYTE)v20 )
    {
      if ( v39 )
        goto LABEL_38;
      goto LABEL_49;
    }
    v21 = *(_QWORD *)(v18 + 16);
    v15 = v39;
    if ( v21 )
      break;
LABEL_33:
    v16 = v33;
    if ( !(_DWORD)v33 )
      goto LABEL_37;
LABEL_34:
    v14 = v32;
  }
  do
  {
    v22 = *(_QWORD *)(v21 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v22 - 30) <= 0xAu )
    {
      while ( 1 )
      {
        v23 = *(__int64 **)(v22 + 40);
        if ( v23 == v7 )
          goto LABEL_30;
        if ( !(_BYTE)v15 )
          break;
        v24 = v36;
        v25 = &v36[8 * HIDWORD(v37)];
        if ( v36 == v25 )
          goto LABEL_42;
        while ( v23 != *(__int64 **)v24 )
        {
          v24 += 8;
          if ( v25 == v24 )
            goto LABEL_42;
        }
LABEL_30:
        v21 = *(_QWORD *)(v21 + 8);
        if ( !v21 )
          goto LABEL_33;
        while ( 1 )
        {
          v22 = *(_QWORD *)(v21 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v22 - 30) <= 0xAu )
            break;
          v21 = *(_QWORD *)(v21 + 8);
          if ( !v21 )
            goto LABEL_33;
        }
      }
      if ( !sub_C8CA60((__int64)&v35, *(_QWORD *)(v22 + 40)) )
      {
LABEL_42:
        v28 = (unsigned int)v33;
        v29 = (unsigned int)v33 + 1LL;
        if ( v29 > HIDWORD(v33) )
        {
          sub_C8D5F0((__int64)&v32, v34, v29, 8u, a5, a6);
          v28 = (unsigned int)v33;
        }
        v32[v28] = v23;
        LODWORD(v33) = v33 + 1;
      }
      v15 = v39;
      goto LABEL_30;
    }
    v21 = *(_QWORD *)(v21 + 8);
  }
  while ( v21 );
  v16 = v33;
  if ( (_DWORD)v33 )
    goto LABEL_34;
LABEL_37:
  if ( (_BYTE)v15 )
  {
LABEL_38:
    v26 = (unsigned __int64)v32;
    if ( v32 == v34 )
      return v13;
    goto LABEL_39;
  }
LABEL_49:
  _libc_free((unsigned __int64)v36);
  v26 = (unsigned __int64)v32;
  if ( v32 != v34 )
LABEL_39:
    _libc_free(v26);
  return v13;
}
