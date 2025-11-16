// Function: sub_1597F80
// Address: 0x1597f80
//
__int64 __fastcall sub_1597F80(__int64 **a1)
{
  __int64 *v1; // rdx
  __int64 result; // rax
  __int64 v3; // r14
  __int64 v4; // r15
  __int64 v5; // r12
  __int64 *v6; // rdx
  int v7; // r9d
  _BYTE *v8; // r8
  unsigned int v10; // r9d
  __int64 v11; // rbx
  _BYTE *v12; // rdx
  __int64 *v13; // r10
  __int64 v14; // rax
  __int64 *v15; // rdi
  _BYTE *v16; // rsi
  unsigned int v17; // ebx
  int v18; // r15d
  __int64 **v19; // rcx
  int v20; // edx
  int v21; // esi
  __int64 *v22; // [rsp+0h] [rbp-190h]
  unsigned int v23; // [rsp+Ch] [rbp-184h]
  _BYTE *v24; // [rsp+10h] [rbp-180h]
  _BYTE *v25; // [rsp+18h] [rbp-178h]
  int v26; // [rsp+2Ch] [rbp-164h] BYREF
  __int64 v27[4]; // [rsp+30h] [rbp-160h] BYREF
  _BYTE *v28; // [rsp+50h] [rbp-140h] BYREF
  __int64 v29; // [rsp+58h] [rbp-138h]
  _BYTE v30[304]; // [rsp+60h] [rbp-130h] BYREF

  v1 = *a1;
  result = **a1;
  v3 = *(_QWORD *)result;
  v4 = *(unsigned int *)(*(_QWORD *)result + 1640LL);
  v5 = *(_QWORD *)(*(_QWORD *)result + 1624LL);
  if ( !(_DWORD)v4 )
  {
LABEL_2:
    v6 = (__int64 *)(v5 + 8 * v4);
    goto LABEL_3;
  }
  v7 = *((_DWORD *)a1 + 5);
  v8 = v30;
  v28 = v30;
  v29 = 0x2000000000LL;
  v10 = v7 & 0xFFFFFFF;
  if ( v10 )
  {
    v11 = 1;
    v12 = v30;
    v13 = a1[-3 * v10];
    v14 = 0;
    while ( 1 )
    {
      *(_QWORD *)&v12[8 * v14] = v13;
      v14 = (unsigned int)(v29 + 1);
      LODWORD(v29) = v29 + 1;
      if ( v10 == (_DWORD)v11 )
        break;
      v13 = a1[3 * (v11 - (*((_DWORD *)a1 + 5) & 0xFFFFFFF))];
      if ( HIDWORD(v29) <= (unsigned int)v14 )
      {
        v22 = a1[3 * (v11 - (*((_DWORD *)a1 + 5) & 0xFFFFFFF))];
        v23 = v10;
        v24 = v8;
        sub_16CD150(&v28, v8, 0, 8);
        v14 = (unsigned int)v29;
        v13 = v22;
        v10 = v23;
        v8 = v24;
      }
      v12 = v28;
      ++v11;
    }
    v15 = (__int64 *)v28;
    v1 = *a1;
    v16 = &v28[8 * v14];
  }
  else
  {
    v16 = v30;
    v14 = 0;
    v15 = (__int64 *)v30;
  }
  v25 = v8;
  v27[0] = (__int64)v1;
  v27[1] = (__int64)v15;
  v27[2] = v14;
  v26 = sub_1597240(v15, (__int64)v16);
  v17 = sub_1597ED0(v27, &v26);
  if ( v28 != v25 )
    _libc_free((unsigned __int64)v28);
  v18 = v4 - 1;
  result = v18 & v17;
  v6 = (__int64 *)(v5 + 8 * result);
  v19 = (__int64 **)*v6;
  if ( a1 != (__int64 **)*v6 )
  {
    v20 = 1;
    while ( v19 != (__int64 **)-8LL )
    {
      v21 = v20 + 1;
      result = v18 & (unsigned int)(v20 + result);
      v6 = (__int64 *)(v5 + 8LL * (unsigned int)result);
      v19 = (__int64 **)*v6;
      if ( a1 == (__int64 **)*v6 )
        goto LABEL_3;
      v20 = v21;
    }
    v5 = *(_QWORD *)(v3 + 1624);
    v4 = *(unsigned int *)(v3 + 1640);
    goto LABEL_2;
  }
LABEL_3:
  *v6 = -16;
  --*(_DWORD *)(v3 + 1632);
  ++*(_DWORD *)(v3 + 1636);
  return result;
}
