// Function: sub_18BA270
// Address: 0x18ba270
//
_QWORD *__fastcall sub_18BA270(__int64 a1, __int64 a2, unsigned __int64 a3, int a4, __int64 *a5, _QWORD *a6)
{
  unsigned __int64 v6; // r10
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rsi
  _QWORD *result; // rax
  __int64 v11; // r12
  unsigned int v12; // r15d
  int v13; // r9d
  _QWORD *v14; // rdx
  unsigned __int64 v15; // rbx
  __int64 v16; // rsi
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // r13
  unsigned __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // rdi
  int v24; // edx
  __int64 v25; // rdi
  unsigned int v26; // edx
  int v27; // ecx
  unsigned __int64 v28; // r8
  unsigned __int64 v30; // r11
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // r11
  unsigned __int64 v33; // [rsp+8h] [rbp-58h]
  int v34; // [rsp+8h] [rbp-58h]
  int v35; // [rsp+8h] [rbp-58h]
  unsigned __int64 v36; // [rsp+10h] [rbp-50h]
  unsigned __int64 v37; // [rsp+10h] [rbp-50h]
  unsigned __int64 v38; // [rsp+10h] [rbp-50h]
  unsigned __int64 v39; // [rsp+18h] [rbp-48h]
  unsigned __int64 v40; // [rsp+18h] [rbp-48h]
  unsigned __int64 v41; // [rsp+18h] [rbp-48h]
  _QWORD *v42; // [rsp+20h] [rbp-40h]
  _QWORD *v43; // [rsp+20h] [rbp-40h]
  _QWORD *v44; // [rsp+20h] [rbp-40h]
  __int64 v45; // [rsp+28h] [rbp-38h]

  v6 = a3;
  v7 = a1;
  if ( a4 == 1 )
    v8 = ~(a3 >> 3);
  else
    v8 = -(__int64)(((unsigned int)(a4 + 7) >> 3) + ((a3 + 7) >> 3));
  *a5 = v8;
  v9 = 32 * a2;
  result = (_QWORD *)(a3 & 7);
  *a6 = result;
  if ( a1 != a1 + v9 )
  {
    v11 = a1 + v9;
    v45 = (unsigned __int8)((unsigned int)(a4 + 7) >> 3);
    v12 = (unsigned __int8)((unsigned int)(a4 + 7) >> 3);
    v13 = a4;
    while ( 1 )
    {
      while ( 1 )
      {
        v14 = *(_QWORD **)(v7 + 8);
        v15 = *(_QWORD *)(v7 + 16);
        result = (_QWORD *)*v14;
        v16 = *(_QWORD *)(*v14 + 16LL);
        v17 = v6 - 8LL * v14[1];
        v18 = *(_QWORD *)(*v14 + 24LL) - v16;
        v19 = v17 >> 3;
        if ( v13 != 1 )
          break;
        if ( v19 + 1 > v18 )
        {
          v33 = v6;
          v36 = v6 - 8LL * v14[1];
          v39 = v19 + 1;
          v42 = (_QWORD *)*v14;
          sub_CD93F0(result + 2, v19 + 1 - v18);
          LOBYTE(v17) = v36;
          v6 = v33;
          v23 = v42[5];
          v13 = 1;
          v30 = v42[6] - v23;
          if ( v19 + 1 > v30 )
          {
            sub_CD93F0(v42 + 5, v39 - v30);
            LOBYTE(v17) = v36;
            v6 = v33;
            v13 = 1;
            v23 = v42[5];
            v16 = v42[2];
          }
          else
          {
            if ( v39 < v30 && v42[6] != v23 + v39 )
              v42[6] = v23 + v39;
            v16 = v42[2];
          }
        }
        else
        {
          v23 = result[5];
        }
        result = (_QWORD *)(v23 + v19);
        v24 = 1 << (v17 & 7);
        if ( v15 )
          *(_BYTE *)(v16 + v19) |= v24;
        v7 += 32;
        *(_BYTE *)result |= v24;
        if ( v7 == v11 )
          return result;
      }
      v20 = v45 + v19;
      if ( *(_BYTE *)(v7 + 24) )
      {
        if ( v20 > v18 )
        {
          v34 = v13;
          v37 = v6;
          v40 = v45 + v19;
          v43 = result;
          sub_CD93F0(result + 2, v20 - v18);
          result = v43;
          v6 = v37;
          v13 = v34;
          v21 = v43[5];
          v31 = v43[6] - v21;
          if ( v31 < v45 + v19 )
          {
            sub_CD93F0(v43 + 5, v40 - v31);
            result = v43;
            v6 = v37;
            v13 = v34;
            v21 = v43[5];
            v16 = v43[2];
          }
          else
          {
            if ( v31 > v40 && v43[6] != v21 + v40 )
              v43[6] = v21 + v40;
            v16 = v43[2];
          }
        }
        else
        {
          v21 = result[5];
        }
        if ( v12 )
        {
          result = 0;
          v22 = v19 + v16;
          do
          {
            *((_BYTE *)result + v22) = v15 >> (8 * (unsigned __int8)result);
            *((_BYTE *)result + v21 + v19) = -1;
            result = (_QWORD *)((char *)result + 1);
          }
          while ( result != (_QWORD *)v12 );
        }
LABEL_12:
        v7 += 32;
        if ( v7 == v11 )
          return result;
      }
      else
      {
        if ( v20 > v18 )
        {
          v35 = v13;
          v38 = v6;
          v41 = v45 + v19;
          v44 = result;
          sub_CD93F0(result + 2, v20 - v18);
          result = v44;
          v6 = v38;
          v13 = v35;
          v25 = v44[5];
          v32 = v44[6] - v25;
          if ( v32 < v45 + v19 )
          {
            sub_CD93F0(v44 + 5, v41 - v32);
            result = v44;
            v6 = v38;
            v13 = v35;
            v25 = v44[5];
            v16 = v44[2];
          }
          else
          {
            v16 = v44[2];
            if ( v32 > v41 && v44[6] != v25 + v41 )
              v44[6] = v25 + v41;
          }
        }
        else
        {
          v25 = result[5];
        }
        v26 = v12 - 1;
        v27 = 0;
        if ( !v12 )
          goto LABEL_12;
        do
        {
          result = (_QWORD *)(v19 + v26);
          v28 = v15 >> v27;
          v27 += 8;
          *((_BYTE *)result + v16) = v28;
          *((_BYTE *)result + v25) = -1;
        }
        while ( v26-- != 0 );
        v7 += 32;
        if ( v7 == v11 )
          return result;
      }
    }
  }
  return result;
}
