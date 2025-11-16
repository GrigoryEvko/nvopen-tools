// Function: sub_2831D20
// Address: 0x2831d20
//
unsigned int *__fastcall sub_2831D20(unsigned int **a1)
{
  __int64 v1; // rcx
  unsigned int *result; // rax
  _QWORD *v4; // r12
  __int64 v5; // rax
  char v6; // dh
  __int64 v7; // rsi
  char v8; // al
  __int64 v9; // rcx
  int *v10; // rdx
  _QWORD *v11; // r13
  int v12; // eax
  __int64 v13; // r15
  __int64 v14; // r13
  unsigned int *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rsi
  unsigned int **v19; // rax
  unsigned int **v20; // rcx
  unsigned int *v21; // rdx
  unsigned int *v22; // rax
  _QWORD *v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rcx
  _QWORD *v27; // r15
  _BYTE *v28; // rax
  unsigned int *v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // rdx
  int v32; // ecx
  __int64 v33; // r8
  int v34; // ecx
  unsigned int v35; // edx
  __int64 *v36; // rax
  __int64 v37; // r10
  __int64 v38; // rax
  unsigned int *v39; // rax
  _QWORD *v40; // rsi
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  int v45; // eax
  int v46; // r9d
  __int64 v47; // [rsp-58h] [rbp-58h]
  __int64 v48; // [rsp-50h] [rbp-50h]
  __int64 v49[8]; // [rsp-40h] [rbp-40h] BYREF

  v1 = **a1;
  result = a1[1];
  if ( result[10] > (unsigned int)v1 )
  {
    while ( 1 )
    {
      v4 = (_QWORD *)sub_B47F80(*(_BYTE **)(*((_QWORD *)result + 4) + 8 * v1));
      v5 = sub_AA4FF0((__int64)a1[4]);
      v9 = v47;
      v7 = v5;
      v8 = 0;
      LOBYTE(v9) = 1;
      if ( v7 )
        v8 = v6;
      BYTE1(v9) = v8;
      v47 = v9;
      sub_B44220(v4, v7, v9);
      v10 = (int *)*a1;
      v11 = *(_QWORD **)(*((_QWORD *)a1[1] + 4) + 8LL * **a1);
      v12 = **a1;
      v13 = v11[2];
      if ( v13 )
      {
        while ( 1 )
        {
          v14 = v13;
          v15 = a1[2];
          v13 = *(_QWORD *)(v13 + 8);
          v16 = *(_QWORD *)(v14 + 24);
          v49[0] = v16;
          v17 = *((_QWORD *)v15 + 1);
          v18 = *(_QWORD *)(v16 + 40);
          if ( *(_BYTE *)(v17 + 84) )
          {
            v19 = *(unsigned int ***)(v17 + 64);
            v20 = &v19[*(unsigned int *)(v17 + 76)];
            if ( v19 == v20 )
              goto LABEL_12;
            while ( 1 )
            {
              v21 = *v19;
              if ( (unsigned int *)v18 == *v19 )
                break;
              if ( v20 == ++v19 )
                goto LABEL_12;
            }
          }
          else
          {
            v48 = v16;
            if ( !sub_C8CA60(v17 + 56, v18) )
              goto LABEL_12;
            v21 = *(unsigned int **)(v48 + 40);
          }
          if ( a1[4] == v21
            || (v22 = a1[3],
                v23 = (_QWORD *)(*(_QWORD *)v22 + 8LL * v22[2]),
                v23 != sub_282F9C0(*(_QWORD **)v22, (__int64)v23, v49)) )
          {
LABEL_12:
            if ( *(_QWORD *)v14 )
            {
              v24 = *(_QWORD *)(v14 + 8);
              **(_QWORD **)(v14 + 16) = v24;
              if ( v24 )
                *(_QWORD *)(v24 + 16) = *(_QWORD *)(v14 + 16);
            }
            *(_QWORD *)v14 = v4;
            if ( v4 )
            {
              v25 = v4[2];
              *(_QWORD *)(v14 + 8) = v25;
              if ( v25 )
                *(_QWORD *)(v25 + 16) = v14 + 8;
              *(_QWORD *)(v14 + 16) = v4 + 2;
              v4[2] = v14;
            }
          }
          if ( !v13 )
          {
            v10 = (int *)*a1;
            v11 = *(_QWORD **)(*((_QWORD *)a1[1] + 4) + 8LL * **a1);
            v12 = **a1;
            break;
          }
        }
      }
      v26 = 32LL * (*((_DWORD *)v11 + 1) & 0x7FFFFFF);
      if ( (*((_BYTE *)v11 + 7) & 0x40) != 0 )
      {
        v27 = (_QWORD *)*(v11 - 1);
        v11 = &v27[(unsigned __int64)v26 / 8];
      }
      else
      {
        v27 = &v11[v26 / 0xFFFFFFFFFFFFFFF8LL];
      }
      if ( v27 != v11 )
        break;
LABEL_34:
      *v10 = v12 + 1;
      v1 = **a1;
      result = a1[1];
      if ( (unsigned int)v1 >= result[10] )
        return result;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v28 = (_BYTE *)*v27;
        if ( *(_BYTE *)*v27 > 0x1Cu )
          break;
LABEL_25:
        v27 += 4;
        if ( v11 == v27 )
          goto LABEL_33;
      }
      v29 = a1[2];
      v49[0] = *v27;
      v30 = *((_QWORD *)v28 + 5);
      v31 = *((_QWORD *)v29 + 3);
      v32 = *(_DWORD *)(v31 + 24);
      v33 = *(_QWORD *)(v31 + 8);
      if ( !v32 )
        goto LABEL_40;
      v34 = v32 - 1;
      v35 = v34 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v36 = (__int64 *)(v33 + 16LL * v35);
      v37 = *v36;
      if ( v30 != *v36 )
        break;
LABEL_29:
      v38 = v36[1];
LABEL_30:
      if ( *((_QWORD *)v29 + 1) != v38 )
        goto LABEL_25;
      v39 = a1[3];
      v40 = (_QWORD *)(*(_QWORD *)v39 + 8LL * v39[2]);
      if ( v40 != sub_282F9C0(*(_QWORD **)v39, (__int64)v40, v49) )
        goto LABEL_25;
      v27 += 4;
      sub_28316E0((__int64)a1[1], v49, v41, v42, v43, v44);
      if ( v11 == v27 )
      {
LABEL_33:
        v10 = (int *)*a1;
        v12 = **a1;
        goto LABEL_34;
      }
    }
    v45 = 1;
    while ( v37 != -4096 )
    {
      v46 = v45 + 1;
      v35 = v34 & (v45 + v35);
      v36 = (__int64 *)(v33 + 16LL * v35);
      v37 = *v36;
      if ( v30 == *v36 )
        goto LABEL_29;
      v45 = v46;
    }
LABEL_40:
    v38 = 0;
    goto LABEL_30;
  }
  return result;
}
