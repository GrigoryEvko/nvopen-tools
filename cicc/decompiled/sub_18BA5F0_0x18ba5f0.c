// Function: sub_18BA5F0
// Address: 0x18ba5f0
//
_QWORD *__fastcall sub_18BA5F0(__int64 a1, __int64 a2, unsigned __int64 a3, int a4, unsigned __int64 *a5, _QWORD *a6)
{
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // rsi
  _QWORD *result; // rax
  int v11; // r10d
  __int64 v12; // r13
  __int64 v13; // r12
  unsigned __int64 v14; // rax
  unsigned int v15; // r15d
  unsigned __int64 v16; // r9
  _QWORD *v17; // rcx
  unsigned __int64 v18; // rbx
  __int64 v19; // rsi
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // r14
  unsigned __int64 v23; // rdx
  __int64 v24; // rdi
  unsigned int v25; // edx
  int v26; // ecx
  unsigned __int64 v27; // r8
  __int64 v29; // rdi
  int v30; // edx
  __int64 v31; // rcx
  __int64 v32; // rsi
  unsigned __int64 v33; // r11
  unsigned __int64 v34; // r11
  unsigned __int64 v35; // r11
  unsigned __int64 v36; // [rsp+8h] [rbp-58h]
  int v37; // [rsp+8h] [rbp-58h]
  int v38; // [rsp+8h] [rbp-58h]
  char v39; // [rsp+10h] [rbp-50h]
  unsigned __int64 v40; // [rsp+10h] [rbp-50h]
  unsigned __int64 v41; // [rsp+10h] [rbp-50h]
  unsigned __int64 v42; // [rsp+18h] [rbp-48h]
  unsigned __int64 v43; // [rsp+18h] [rbp-48h]
  unsigned __int64 v44; // [rsp+18h] [rbp-48h]
  _QWORD *v45; // [rsp+20h] [rbp-40h]
  _QWORD *v46; // [rsp+20h] [rbp-40h]
  _QWORD *v47; // [rsp+20h] [rbp-40h]
  __int64 v48; // [rsp+28h] [rbp-38h]

  v7 = a3 >> 3;
  v8 = (a3 + 7) >> 3;
  if ( a4 == 1 )
    v8 = v7;
  v9 = 32 * a2;
  *a5 = v8;
  result = (_QWORD *)(a3 & 7);
  *a6 = result;
  if ( a1 != a1 + v9 )
  {
    v11 = a4;
    v12 = a1;
    v13 = a1 + v9;
    v48 = (unsigned __int8)((unsigned int)(a4 + 7) >> 3);
    v14 = a3;
    v15 = (unsigned __int8)((unsigned int)(a4 + 7) >> 3);
    v16 = v14;
    while ( 1 )
    {
      while ( 1 )
      {
        v17 = *(_QWORD **)(v12 + 8);
        v18 = *(_QWORD *)(v12 + 16);
        result = (_QWORD *)*v17;
        v19 = *(_QWORD *)(*v17 + 64LL);
        v20 = v16 - 8LL * (*(_QWORD *)(*v17 + 8LL) - v17[1]);
        v21 = result[9] - v19;
        v22 = v20 >> 3;
        if ( v11 != 1 )
          break;
        if ( v22 + 1 > v21 )
        {
          v36 = v16;
          v39 = v20;
          v42 = v22 + 1;
          v45 = result;
          sub_CD93F0(result + 8, v22 + 1 - v21);
          LOBYTE(v20) = v39;
          v16 = v36;
          v29 = v45[11];
          v11 = 1;
          v33 = v45[12] - v29;
          if ( v22 + 1 > v33 )
          {
            sub_CD93F0(v45 + 11, v42 - v33);
            LOBYTE(v20) = v39;
            v16 = v36;
            v11 = 1;
            v29 = v45[11];
            v19 = v45[8];
          }
          else
          {
            if ( v42 < v33 && v45[12] != v29 + v42 )
              v45[12] = v29 + v42;
            v19 = v45[8];
          }
        }
        else
        {
          v29 = result[11];
        }
        result = (_QWORD *)(v29 + v22);
        v30 = 1 << (v20 & 7);
        if ( v18 )
          *(_BYTE *)(v19 + v22) |= v30;
        v12 += 32;
        *(_BYTE *)result |= v30;
        if ( v12 == v13 )
          return result;
      }
      v23 = v48 + v22;
      if ( *(_BYTE *)(v12 + 24) )
      {
        if ( v23 > v21 )
        {
          v37 = v11;
          v40 = v16;
          v43 = v48 + v22;
          v46 = result;
          sub_CD93F0(result + 8, v23 - v21);
          result = v46;
          v16 = v40;
          v11 = v37;
          v24 = v46[11];
          v34 = v46[12] - v24;
          if ( v34 < v48 + v22 )
          {
            sub_CD93F0(v46 + 11, v43 - v34);
            result = v46;
            v16 = v40;
            v11 = v37;
            v24 = v46[11];
            v19 = v46[8];
          }
          else
          {
            if ( v34 > v43 && v46[12] != v24 + v43 )
              v46[12] = v24 + v43;
            v19 = v46[8];
          }
        }
        else
        {
          v24 = result[11];
        }
        v25 = v15 - 1;
        v26 = 0;
        if ( v15 )
        {
          do
          {
            result = (_QWORD *)(v22 + v25);
            v27 = v18 >> v26;
            v26 += 8;
            *((_BYTE *)result + v19) = v27;
            *((_BYTE *)result + v24) = -1;
          }
          while ( v25-- != 0 );
        }
LABEL_11:
        v12 += 32;
        if ( v12 == v13 )
          return result;
      }
      else
      {
        if ( v23 > v21 )
        {
          v38 = v11;
          v41 = v16;
          v44 = v48 + v22;
          v47 = result;
          sub_CD93F0(result + 8, v23 - v21);
          result = v47;
          v16 = v41;
          v11 = v38;
          v31 = v47[11];
          v35 = v47[12] - v31;
          if ( v35 < v48 + v22 )
          {
            sub_CD93F0(v47 + 11, v44 - v35);
            result = v47;
            v16 = v41;
            v11 = v38;
            v31 = v47[11];
            v19 = v47[8];
          }
          else
          {
            v19 = v47[8];
            if ( v35 > v44 && v47[12] != v31 + v44 )
              v47[12] = v31 + v44;
          }
        }
        else
        {
          v31 = result[11];
        }
        if ( !v15 )
          goto LABEL_11;
        result = 0;
        v32 = v22 + v19;
        do
        {
          *((_BYTE *)result + v32) = v18 >> (8 * (unsigned __int8)result);
          *((_BYTE *)result + v31 + v22) = -1;
          result = (_QWORD *)((char *)result + 1);
        }
        while ( (_QWORD *)v15 != result );
        v12 += 32;
        if ( v12 == v13 )
          return result;
      }
    }
  }
  return result;
}
