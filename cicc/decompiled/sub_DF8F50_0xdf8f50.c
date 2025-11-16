// Function: sub_DF8F50
// Address: 0xdf8f50
//
__int64 __fastcall sub_DF8F50(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, char a5, char a6)
{
  _QWORD *v8; // rsi
  __int64 v9; // rdi
  __int64 *v10; // rdi
  __int64 *v11; // r14
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rbx
  __int64 v16; // rax
  int v17; // ecx
  __int64 v18; // rdi
  int v19; // ecx
  unsigned int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // r10
  _QWORD *v23; // rax
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // r12
  __int64 v29; // rdx
  __int64 v30; // rbx
  __int64 v31; // r13
  _QWORD *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // rax
  unsigned __int64 v36; // rax
  int v37; // eax
  int v38; // r8d
  __int64 v39; // rax
  __int64 v41; // [rsp+18h] [rbp-88h]
  __int64 *v45; // [rsp+30h] [rbp-70h]
  int v46; // [rsp+38h] [rbp-68h]
  __int64 v47; // [rsp+38h] [rbp-68h]
  __int64 *v48; // [rsp+40h] [rbp-60h] BYREF
  __int64 v49; // [rsp+48h] [rbp-58h]
  _BYTE v50[80]; // [rsp+50h] [rbp-50h] BYREF

  v8 = &v48;
  v9 = *(_QWORD *)a1;
  v48 = (__int64 *)v50;
  v49 = 0x400000000LL;
  sub_D46D90(v9, (__int64)&v48);
  v10 = v48;
  v45 = &v48[(unsigned int)v49];
  if ( v48 != v45 )
  {
    v11 = v48;
    while ( 1 )
    {
      v8 = *(_QWORD **)a1;
      v12 = *v11;
      v13 = *(_QWORD *)(**(_QWORD **)(*(_QWORD *)a1 + 32LL) + 16LL);
      if ( v13 )
      {
        while ( 1 )
        {
          v14 = *(_QWORD *)(v13 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v14 - 30) <= 0xAu )
            break;
          v13 = *(_QWORD *)(v13 + 8);
          if ( !v13 )
            goto LABEL_9;
        }
LABEL_7:
        if ( v12 == *(_QWORD *)(v14 + 40) )
          goto LABEL_11;
        while ( 1 )
        {
          v13 = *(_QWORD *)(v13 + 8);
          if ( !v13 )
            break;
          v14 = *(_QWORD *)(v13 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v14 - 30) <= 0xAu )
            goto LABEL_7;
        }
      }
LABEL_9:
      if ( a6 || a1[49] )
        goto LABEL_22;
LABEL_11:
      v15 = sub_DBA6E0(a2, (__int64)v8, *v11, 0);
      if ( sub_D96A50(v15) )
        goto LABEL_22;
      if ( *(_WORD *)(v15 + 24) )
      {
        v8 = (_QWORD *)v15;
        if ( !sub_DADE90(a2, v15, *(_QWORD *)a1) )
          goto LABEL_22;
      }
      else
      {
        v16 = *(_QWORD *)(v15 + 32);
        if ( *(_DWORD *)(v16 + 32) <= 0x40u )
        {
          if ( !*(_QWORD *)(v16 + 24) )
            goto LABEL_22;
        }
        else
        {
          v46 = *(_DWORD *)(v16 + 32);
          if ( v46 == (unsigned int)sub_C444A0(v16 + 24) )
            goto LABEL_22;
        }
      }
      v8 = (_QWORD *)sub_D95540(v15);
      if ( sub_D97050(a2, (__int64)v8) > (unsigned __int64)(*(_DWORD *)(*((_QWORD *)a1 + 4) + 8LL) >> 8) )
        goto LABEL_22;
      v8 = *(_QWORD **)a1;
      if ( !a1[48] )
      {
        v17 = *(_DWORD *)(a3 + 24);
        v18 = *(_QWORD *)(a3 + 8);
        if ( v17 )
        {
          v19 = v17 - 1;
          v20 = v19 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v21 = (__int64 *)(v18 + 16LL * v20);
          v22 = *v21;
          if ( v12 == *v21 )
          {
LABEL_19:
            v23 = (_QWORD *)v21[1];
LABEL_20:
            if ( v23 != v8 && !a5 )
              goto LABEL_22;
            goto LABEL_31;
          }
          v37 = 1;
          while ( v22 != -4096 )
          {
            v38 = v37 + 1;
            v39 = v19 & (v20 + v37);
            v20 = v39;
            v21 = (__int64 *)(v18 + 16 * v39);
            v22 = *v21;
            if ( v12 == *v21 )
              goto LABEL_19;
            v37 = v38;
          }
        }
        v23 = 0;
        goto LABEL_20;
      }
LABEL_31:
      v25 = *(_QWORD *)(*(_QWORD *)v8[4] + 16LL);
      if ( v25 )
      {
        while ( 1 )
        {
          v26 = *(_QWORD *)(v25 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v26 - 30) <= 0xAu )
            break;
          v25 = *(_QWORD *)(v25 + 8);
          if ( !v25 )
            goto LABEL_46;
        }
        v27 = a2;
        v47 = v15;
        v28 = v12;
        v29 = *(_QWORD *)(v26 + 40);
        v30 = v25;
        v31 = v27;
        if ( *((_BYTE *)v8 + 84) )
          goto LABEL_34;
LABEL_42:
        v34 = (__int64)(v8 + 7);
        v8 = (_QWORD *)v29;
        v41 = v29;
        if ( sub_C8CA60(v34, v29) )
        {
          v8 = (_QWORD *)v28;
          if ( !(unsigned __int8)sub_B19720(a4, v28, v41) )
          {
LABEL_58:
            a2 = v31;
            goto LABEL_22;
          }
        }
LABEL_39:
        while ( 1 )
        {
          v30 = *(_QWORD *)(v30 + 8);
          if ( !v30 )
            break;
          v33 = *(_QWORD *)(v30 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v33 - 30) <= 0xAu )
          {
            v8 = *(_QWORD **)a1;
            v29 = *(_QWORD *)(v33 + 40);
            if ( !*(_BYTE *)(*(_QWORD *)a1 + 84LL) )
              goto LABEL_42;
LABEL_34:
            v32 = (_QWORD *)v8[8];
            v8 = &v32[*((unsigned int *)v8 + 19)];
            if ( v32 == v8 )
              continue;
            while ( v29 != *v32 )
            {
              if ( v8 == ++v32 )
                goto LABEL_39;
            }
            v8 = (_QWORD *)v28;
            if ( (unsigned __int8)sub_B19720(a4, v28, v29) )
              continue;
            goto LABEL_58;
          }
        }
        v35 = v31;
        v15 = v47;
        v12 = v28;
        a2 = v35;
      }
LABEL_46:
      v36 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v36 != v12 + 48 )
      {
        if ( !v36 )
          BUG();
        if ( *(_BYTE *)(v36 - 24) == 31 && (*(_DWORD *)(v36 - 20) & 0x7FFFFFF) == 3 )
        {
          *((_QWORD *)a1 + 1) = v12;
          v10 = v48;
          *((_QWORD *)a1 + 2) = v36 - 24;
          *((_QWORD *)a1 + 3) = v15;
          goto LABEL_24;
        }
      }
LABEL_22:
      if ( v45 == ++v11 )
      {
        v12 = *((_QWORD *)a1 + 1);
        v10 = v48;
        goto LABEL_24;
      }
    }
  }
  v12 = *((_QWORD *)a1 + 1);
LABEL_24:
  if ( v10 != (__int64 *)v50 )
    _libc_free(v10, v8);
  LOBYTE(a2) = v12 != 0;
  return (unsigned int)a2;
}
