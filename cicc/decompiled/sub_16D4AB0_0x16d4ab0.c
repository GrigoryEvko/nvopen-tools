// Function: sub_16D4AB0
// Address: 0x16d4ab0
//
unsigned __int64 __fastcall sub_16D4AB0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // r14
  int v4; // r13d
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rax
  unsigned __int64 result; // rax
  __int64 v13; // r12
  char *v14; // rbx
  __int64 v15; // rax
  _QWORD *v16; // rax
  int v17; // ebx
  _QWORD *v18; // rax
  _QWORD *v19; // r14
  _QWORD *v20; // r10
  __int64 v21; // r11
  __int64 v22; // r12
  __int64 v23; // rax
  bool v24; // cf
  unsigned __int64 v25; // r12
  __int64 v26; // r12
  _QWORD *v27; // r11
  _QWORD *v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // r14
  __int64 v31; // rax
  __int64 v32; // rax
  _QWORD *v33; // [rsp+8h] [rbp-58h]
  char *v34; // [rsp+8h] [rbp-58h]
  _QWORD *v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  _QWORD v38[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = (unsigned int)a2;
  v4 = (int)a2;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 32) = 8;
  v5 = sub_22077B0(64);
  v6 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 24) = v5;
  v7 = (__int64 *)(v5 + ((4 * v6 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v8 = sub_22077B0(512);
  *(_QWORD *)(a1 + 64) = v7;
  *v7 = v8;
  *(_QWORD *)(a1 + 48) = v8;
  *(_QWORD *)(a1 + 80) = v8;
  *(_QWORD *)(a1 + 40) = v8;
  *(_QWORD *)(a1 + 72) = v8;
  *(_QWORD *)(a1 + 56) = v8 + 512;
  *(_QWORD *)(a1 + 88) = v8 + 512;
  *(_QWORD *)(a1 + 96) = v7;
  *(_QWORD *)(a1 + 136) = 0;
  *(_OWORD *)(a1 + 104) = 0;
  *(_OWORD *)(a1 + 120) = 0;
  sub_2210B10(a1 + 144);
  *(_QWORD *)(a1 + 224) = 0;
  *(_OWORD *)(a1 + 192) = 0;
  *(_OWORD *)(a1 + 208) = 0;
  sub_2210B10(a1 + 232);
  v10 = *(_QWORD *)a1;
  v11 = *(_QWORD *)(a1 + 16);
  *(_DWORD *)(a1 + 280) = 0;
  *(_BYTE *)(a1 + 284) = 1;
  result = (v11 - v10) >> 3;
  if ( (unsigned int)a2 > result )
  {
    a2 = *(_QWORD **)(a1 + 8);
    v13 = 0;
    v14 = (char *)a2 - v10;
    if ( v3 )
    {
      v15 = sub_22077B0(8 * v3);
      a2 = *(_QWORD **)(a1 + 8);
      v10 = *(_QWORD *)a1;
      v13 = v15;
    }
    if ( a2 != (_QWORD *)v10 )
    {
      v16 = (_QWORD *)v13;
      do
      {
        if ( v16 )
        {
          v9 = *(_QWORD *)v10;
          *v16 = *(_QWORD *)v10;
          *(_QWORD *)v10 = 0;
        }
        else if ( *(_QWORD *)v10 )
        {
          goto LABEL_10;
        }
        v10 += 8;
        ++v16;
      }
      while ( a2 != (_QWORD *)v10 );
      v10 = *(_QWORD *)a1;
    }
    if ( v10 )
    {
      a2 = (_QWORD *)(*(_QWORD *)(a1 + 16) - v10);
      j_j___libc_free_0(v10, a2);
    }
    result = v13 + 8 * v3;
    *(_QWORD *)a1 = v13;
    *(_QWORD *)(a1 + 8) = &v14[v13];
    *(_QWORD *)(a1 + 16) = result;
  }
  v17 = 0;
  if ( v4 )
  {
    do
    {
      while ( 1 )
      {
        v19 = *(_QWORD **)(a1 + 8);
        if ( v19 == *(_QWORD **)(a1 + 16) )
          break;
        if ( v19 )
        {
          *v19 = 0;
          v18 = (_QWORD *)sub_22077B0(16);
          if ( v18 )
          {
            v18[1] = a1;
            *v18 = off_49850E0;
          }
          a2 = v38;
          v38[0] = v18;
          result = sub_22420C0(v19, v38, &pthread_create);
          v10 = v38[0];
          if ( v38[0] )
            result = (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)v38[0] + 8LL))(v38[0]);
          v19 = *(_QWORD **)(a1 + 8);
        }
        ++v17;
        *(_QWORD *)(a1 + 8) = v19 + 1;
        if ( v4 == v17 )
          return result;
      }
      v20 = *(_QWORD **)a1;
      v21 = (__int64)v19 - *(_QWORD *)a1;
      v22 = v21 >> 3;
      if ( v21 >> 3 == 0xFFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"vector::_M_realloc_insert");
      v23 = 1;
      if ( v22 )
        v23 = ((__int64)v19 - *(_QWORD *)a1) >> 3;
      v24 = __CFADD__(v23, v22);
      v25 = v23 + v22;
      v37 = v25;
      if ( v24 )
      {
        v10 = 0x7FFFFFFFFFFFFFF8LL;
        v37 = 0xFFFFFFFFFFFFFFFLL;
      }
      else
      {
        if ( !v25 )
        {
          v26 = 0;
          goto LABEL_31;
        }
        v32 = 0xFFFFFFFFFFFFFFFLL;
        if ( v25 <= 0xFFFFFFFFFFFFFFFLL )
          v32 = v25;
        v37 = v32;
        v10 = 8 * v32;
      }
      v34 = (char *)v19 - *(_QWORD *)a1;
      v36 = *(_QWORD *)a1;
      v31 = sub_22077B0(v10);
      v20 = (_QWORD *)v36;
      v21 = (__int64)v34;
      v26 = v31;
LABEL_31:
      v27 = (_QWORD *)(v26 + v21);
      if ( v27 )
      {
        *v27 = 0;
        v33 = v20;
        v35 = v27;
        v28 = (_QWORD *)sub_22077B0(16);
        if ( v28 )
        {
          v28[1] = a1;
          *v28 = off_49850E0;
        }
        a2 = v38;
        v38[0] = v28;
        sub_22420C0(v35, v38, &pthread_create);
        v10 = v38[0];
        v20 = v33;
        if ( v38[0] )
        {
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v38[0] + 8LL))(v38[0]);
          v20 = v33;
        }
      }
      v9 = v26;
      if ( v19 != v20 )
      {
        v29 = v20;
        while ( 1 )
        {
          while ( v9 )
          {
            a2 = (_QWORD *)*v29++;
            v9 += 8;
            *(_QWORD *)(v9 - 8) = a2;
            *(v29 - 1) = 0;
            if ( v19 == v29 )
              goto LABEL_42;
          }
          if ( *v29 )
            break;
          ++v29;
          v9 = 8;
          if ( v19 == v29 )
            goto LABEL_42;
        }
LABEL_10:
        sub_2207530(v10, a2, v9);
      }
LABEL_42:
      v30 = v9 + 8;
      if ( v20 )
      {
        v10 = (__int64)v20;
        a2 = (_QWORD *)(*(_QWORD *)(a1 + 16) - (_QWORD)v20);
        j_j___libc_free_0(v20, a2);
      }
      ++v17;
      *(_QWORD *)a1 = v26;
      *(_QWORD *)(a1 + 8) = v30;
      result = v26 + 8 * v37;
      *(_QWORD *)(a1 + 16) = result;
    }
    while ( v4 != v17 );
  }
  return result;
}
