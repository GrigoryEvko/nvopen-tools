// Function: sub_13FA0E0
// Address: 0x13fa0e0
//
void __fastcall sub_13FA0E0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rcx
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // r13
  unsigned int v7; // r12d
  __int64 v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // r15
  _QWORD *v11; // rax
  _QWORD *v12; // r14
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rdi
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 *v19; // rcx
  __int64 v20; // rdi
  __int64 v21; // rdx
  _QWORD *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rax
  _QWORD *v27; // rdx
  __int64 *v28; // [rsp+10h] [rbp-170h]
  __int64 *v30; // [rsp+20h] [rbp-160h]
  __int64 v31; // [rsp+28h] [rbp-158h]
  __int64 v32; // [rsp+30h] [rbp-150h]
  int v33; // [rsp+3Ch] [rbp-144h]
  _BYTE *v34; // [rsp+40h] [rbp-140h] BYREF
  __int64 v35; // [rsp+48h] [rbp-138h]
  _BYTE v36[304]; // [rsp+50h] [rbp-130h] BYREF

  v2 = *(__int64 **)(a1 + 32);
  v34 = v36;
  v35 = 0x2000000000LL;
  v28 = *(__int64 **)(a1 + 40);
  v30 = v2;
  v31 = a1 + 56;
  if ( v28 != v2 )
  {
    while ( 1 )
    {
      v4 = *v30;
      LODWORD(v35) = 0;
      v32 = v4;
      v5 = sub_157EBA0(v4);
      if ( v5 )
      {
        v33 = sub_15F4D60(v5);
        v6 = sub_157EBA0(v32);
        if ( v33 )
          break;
      }
LABEL_28:
      if ( v28 == ++v30 )
      {
        if ( v34 != v36 )
          _libc_free((unsigned __int64)v34);
        return;
      }
    }
    v7 = 0;
    while ( 1 )
    {
      v8 = sub_15F4DF0(v6, v7);
      v9 = *(_QWORD **)(a1 + 72);
      v10 = v8;
      v11 = *(_QWORD **)(a1 + 64);
      if ( v9 == v11 )
      {
        v12 = &v11[*(unsigned int *)(a1 + 84)];
        if ( v11 == v12 )
        {
          v27 = *(_QWORD **)(a1 + 64);
        }
        else
        {
          do
          {
            if ( v10 == *v11 )
              break;
            ++v11;
          }
          while ( v12 != v11 );
          v27 = v12;
        }
      }
      else
      {
        v12 = &v9[*(unsigned int *)(a1 + 80)];
        v11 = (_QWORD *)sub_16CC9F0(v31, v10);
        if ( v10 == *v11 )
        {
          v24 = *(_QWORD *)(a1 + 72);
          if ( v24 == *(_QWORD *)(a1 + 64) )
            v25 = *(unsigned int *)(a1 + 84);
          else
            v25 = *(unsigned int *)(a1 + 80);
          v27 = (_QWORD *)(v24 + 8 * v25);
        }
        else
        {
          v13 = *(_QWORD *)(a1 + 72);
          if ( v13 != *(_QWORD *)(a1 + 64) )
          {
            v11 = (_QWORD *)(v13 + 8LL * *(unsigned int *)(a1 + 80));
            goto LABEL_10;
          }
          v11 = (_QWORD *)(v13 + 8LL * *(unsigned int *)(a1 + 84));
          v27 = v11;
        }
      }
      while ( v27 != v11 && *v11 >= 0xFFFFFFFFFFFFFFFELL )
        ++v11;
LABEL_10:
      if ( v11 != v12 )
        goto LABEL_5;
      v14 = *(_QWORD *)(v10 + 8);
      if ( v14 )
      {
        while ( 1 )
        {
          v15 = sub_1648700(v14);
          if ( (unsigned __int8)(*(_BYTE *)(v15 + 16) - 25) <= 9u )
            break;
          v14 = *(_QWORD *)(v14 + 8);
          if ( !v14 )
            goto LABEL_33;
        }
      }
      else
      {
LABEL_33:
        v15 = sub_1648700(0);
      }
      if ( v32 != *(_QWORD *)(v15 + 40) )
        goto LABEL_5;
      v16 = sub_157EBA0(v32);
      if ( !v16 || (int)sub_15F4D60(v16) <= 2 )
      {
        v26 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v26 >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, a2 + 16, 0, 8);
          v26 = *(unsigned int *)(a2 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v26) = v10;
        ++*(_DWORD *)(a2 + 8);
        goto LABEL_5;
      }
      v17 = v34;
      v18 = 8LL * (unsigned int)v35;
      v19 = (__int64 *)&v34[v18];
      v20 = v18 >> 3;
      v21 = v18 >> 5;
      if ( v21 )
      {
        v22 = &v34[32 * v21];
        while ( v10 != *v17 )
        {
          if ( v10 == v17[1] )
          {
            if ( v19 != v17 + 1 )
              goto LABEL_5;
            goto LABEL_24;
          }
          if ( v10 == v17[2] )
          {
            if ( v19 != v17 + 2 )
              goto LABEL_5;
            goto LABEL_24;
          }
          if ( v10 == v17[3] )
          {
            if ( v19 != v17 + 3 )
              goto LABEL_5;
            goto LABEL_24;
          }
          v17 += 4;
          if ( v22 == v17 )
          {
            v20 = v19 - v17;
            goto LABEL_51;
          }
        }
LABEL_23:
        if ( v19 == v17 )
          goto LABEL_24;
LABEL_5:
        if ( v33 == ++v7 )
          goto LABEL_28;
      }
      else
      {
LABEL_51:
        if ( v20 != 2 )
        {
          if ( v20 != 3 )
          {
            if ( v20 != 1 )
            {
LABEL_24:
              if ( (unsigned int)v35 >= HIDWORD(v35) )
                goto LABEL_56;
              goto LABEL_25;
            }
            if ( v10 == *v17 )
              goto LABEL_23;
            goto LABEL_55;
          }
          if ( v10 == *v17 )
            goto LABEL_23;
          ++v17;
        }
        if ( v10 == *v17 )
          goto LABEL_23;
        if ( v10 == *++v17 )
          goto LABEL_23;
LABEL_55:
        if ( (unsigned int)v35 >= HIDWORD(v35) )
        {
LABEL_56:
          sub_16CD150(&v34, v36, 0, 8);
          v19 = (__int64 *)&v34[8 * (unsigned int)v35];
        }
LABEL_25:
        *v19 = v10;
        LODWORD(v35) = v35 + 1;
        v23 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v23 >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, a2 + 16, 0, 8);
          v23 = *(unsigned int *)(a2 + 8);
        }
        ++v7;
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v23) = v10;
        ++*(_DWORD *)(a2 + 8);
        if ( v33 == v7 )
          goto LABEL_28;
      }
    }
  }
}
