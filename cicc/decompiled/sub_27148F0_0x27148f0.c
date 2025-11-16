// Function: sub_27148F0
// Address: 0x27148f0
//
__int64 __fastcall sub_27148F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v7; // r12d
  __int64 v8; // r14
  _QWORD *v9; // r15
  __int64 v10; // r13
  unsigned __int8 *i; // rdi
  __int64 v13; // rdi
  unsigned __int8 *v14; // rax
  __int64 v15; // rdx
  unsigned __int8 v16; // al
  __int64 v17; // r8
  __int64 v18; // r13
  __int64 v19; // r12
  __int64 v20; // rbx
  unsigned __int64 v21; // rdi
  unsigned __int8 *j; // rdi
  __int64 v23; // rdi
  unsigned __int8 *v24; // rax
  __int64 v25; // rdx
  unsigned __int8 v26; // al
  __int64 v27; // r8
  __int64 v28; // r8
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // r9
  unsigned __int8 v33; // [rsp+Fh] [rbp-61h]
  __int64 v35; // [rsp+18h] [rbp-58h]
  __int64 v36; // [rsp+20h] [rbp-50h]
  __int64 v37; // [rsp+20h] [rbp-50h]
  _QWORD *v39; // [rsp+28h] [rbp-48h]
  __int64 v40; // [rsp+28h] [rbp-48h]
  __int64 v41; // [rsp+28h] [rbp-48h]
  __int64 v42[7]; // [rsp+38h] [rbp-38h] BYREF

  v7 = sub_3108990(a2);
  switch ( v7 )
  {
    case 0u:
    case 1u:
      for ( i = *(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
            ;
            i = *(unsigned __int8 **)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF)) )
      {
        v14 = sub_BD3990(i, a2);
        v13 = 23;
        v8 = (__int64)v14;
        v16 = *v14;
        if ( v16 > 0x1Cu )
        {
          if ( v16 == 85 )
          {
            v17 = *(_QWORD *)(v8 - 32);
            v13 = 21;
            if ( v17 && !*(_BYTE *)v17 && *(_QWORD *)(v17 + 24) == *(_QWORD *)(v8 + 80) )
              v13 = (unsigned int)sub_3108960(*(_QWORD *)(v8 - 32), a2, v15);
          }
          else
          {
            v13 = 2 * (unsigned int)(v16 != 34) + 21;
          }
        }
        if ( !(unsigned __int8)sub_3108CA0(v13) )
          break;
      }
      v42[0] = v8;
      v36 = sub_2713AA0(a5 + 64, v42);
      v33 = sub_271D830(v36);
      if ( v33 )
      {
        v28 = v36;
        if ( v7 != 1 )
        {
          v42[0] = a2;
          v30 = sub_2714480(a4, v42);
          v28 = v36;
          *(_BYTE *)v30 = *(_BYTE *)(v36 + 8);
          *(_BYTE *)(v30 + 1) = *(_BYTE *)(v36 + 9);
          *(_QWORD *)(v30 + 8) = *(_QWORD *)(v36 + 16);
          if ( v36 + 24 != v30 + 16 )
          {
            v40 = v30;
            sub_C8CE00(v30 + 16, v30 + 48, v36 + 24, v31, v36, v32);
            v28 = v36;
            v30 = v40;
          }
          if ( v28 + 72 != v30 + 64 )
          {
            v37 = v28;
            v41 = v30;
            sub_C8CE00(v30 + 64, v30 + 96, v28 + 72, v31, v28, v32);
            v28 = v37;
            v30 = v41;
          }
          *(_BYTE *)(v30 + 112) = *(_BYTE *)(v28 + 120);
        }
        sub_271D520(v28, 0);
        v33 = 0;
      }
      goto LABEL_3;
    case 4u:
      for ( j = *(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
            ;
            j = *(unsigned __int8 **)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF)) )
      {
        v24 = sub_BD3990(j, a2);
        v23 = 23;
        v8 = (__int64)v24;
        v26 = *v24;
        if ( v26 > 0x1Cu )
        {
          if ( v26 == 85 )
          {
            v27 = *(_QWORD *)(v8 - 32);
            v23 = 21;
            if ( v27 && !*(_BYTE *)v27 && *(_QWORD *)(v27 + 24) == *(_QWORD *)(v8 + 80) )
              v23 = (unsigned int)sub_3108960(*(_QWORD *)(v8 - 32), a2, v25);
          }
          else
          {
            v23 = 2 * (unsigned int)(v26 != 34) + 21;
          }
        }
        if ( !(unsigned __int8)sub_3108CA0(v23) )
          break;
      }
      v42[0] = v8;
      v29 = sub_2713AA0(a5 + 64, v42);
      v33 = sub_271D660(v29, a1 + 168, a2);
      goto LABEL_3;
    case 7u:
    case 0x18u:
      return 0;
    case 8u:
      sub_270F800(a5 + 64);
      v18 = *(_QWORD *)(a5 + 96);
      v19 = *(_QWORD *)(a5 + 104);
      v33 = 0;
      if ( v18 == v19 )
        return v33;
      v20 = *(_QWORD *)(a5 + 96);
      break;
    default:
      v33 = 0;
      v8 = 0;
LABEL_3:
      v39 = *(_QWORD **)(a5 + 104);
      if ( v39 != *(_QWORD **)(a5 + 96) )
      {
        v9 = *(_QWORD **)(a5 + 96);
        v10 = a1 + 8;
        do
        {
          if ( v8 != *v9 )
          {
            v35 = *v9;
            if ( !(unsigned __int8)sub_271D8F0(v9 + 1, a2, *v9, v10, v7) )
              sub_271D970(v9 + 1, a3, a2, v35, v10, v7);
          }
          v9 += 17;
        }
        while ( v39 != v9 );
      }
      return v33;
  }
  do
  {
    while ( *(_BYTE *)(v20 + 108) )
    {
      if ( !*(_BYTE *)(v20 + 60) )
        goto LABEL_26;
LABEL_23:
      v20 += 136;
      if ( v19 == v20 )
        goto LABEL_27;
    }
    _libc_free(*(_QWORD *)(v20 + 88));
    if ( *(_BYTE *)(v20 + 60) )
      goto LABEL_23;
LABEL_26:
    v21 = *(_QWORD *)(v20 + 40);
    v20 += 136;
    _libc_free(v21);
  }
  while ( v19 != v20 );
LABEL_27:
  *(_QWORD *)(a5 + 104) = v18;
  return 0;
}
