// Function: sub_210BC20
// Address: 0x210bc20
//
double *__fastcall sub_210BC20(_QWORD *a1)
{
  double *result; // rax
  __int64 v2; // rcx
  int v3; // r12d
  int v4; // ebx
  unsigned int v5; // edx
  __int64 v6; // r14
  __int64 v7; // r8
  __int64 (__fastcall *v8)(_QWORD *); // r9
  unsigned __int64 v9; // rax
  unsigned int v10; // r10d
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rsi
  _QWORD *v14; // rdx
  _QWORD *v15; // rax
  unsigned int v16; // [rsp+4h] [rbp-5Ch]
  __int64 (__fastcall *v17)(_QWORD *); // [rsp+8h] [rbp-58h]
  __int64 (__fastcall *v18)(_QWORD *); // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  _QWORD *v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+18h] [rbp-48h]
  int v22[2]; // [rsp+28h] [rbp-38h] BYREF

  result = sub_16D8B50(
             (__m128i **)v22,
             (unsigned __int8 *)"seed",
             4u,
             (__int64)"Seed Live Regs",
             14,
             unk_4F9E388,
             "regalloc",
             8u,
             "Register Allocation",
             (double *)0x13);
  v2 = a1[2];
  v3 = *(_DWORD *)(v2 + 32);
  if ( v3 )
  {
    v4 = 0;
    while ( 1 )
    {
      v5 = v4 & 0x7FFFFFFF;
      v6 = v4 & 0x7FFFFFFF;
      result = *(double **)(*(_QWORD *)(v2 + 24) + 16 * v6 + 8);
      if ( !result )
        goto LABEL_5;
      if ( (*((_BYTE *)result + 4) & 8) == 0 )
        break;
      while ( 1 )
      {
        result = (double *)*((_QWORD *)result + 4);
        if ( !result )
          break;
        if ( (*((_BYTE *)result + 4) & 8) == 0 )
          goto LABEL_9;
      }
      if ( ++v4 == v3 )
        goto LABEL_16;
LABEL_6:
      v2 = a1[2];
    }
LABEL_9:
    v7 = a1[4];
    v8 = *(__int64 (__fastcall **)(_QWORD *))(*a1 + 40LL);
    v9 = *(unsigned int *)(v7 + 408);
    if ( v5 < (unsigned int)v9 && *(_QWORD *)(*(_QWORD *)(v7 + 400) + 8 * v6) )
    {
      result = (double *)v8(a1);
      goto LABEL_5;
    }
    v10 = v5 + 1;
    if ( (unsigned int)v9 < v5 + 1 )
    {
      v12 = v10;
      if ( v10 >= v9 )
      {
        if ( v10 > v9 )
        {
          if ( v10 > (unsigned __int64)*(unsigned int *)(v7 + 412) )
          {
            v16 = v10;
            v17 = *(__int64 (__fastcall **)(_QWORD *))(*a1 + 40LL);
            v19 = a1[4];
            v21 = v10;
            sub_16CD150(v7 + 400, (const void *)(v7 + 416), v10, 8, v7, (int)v8);
            v7 = v19;
            v10 = v16;
            v8 = v17;
            v12 = v21;
            v9 = *(unsigned int *)(v19 + 408);
          }
          v11 = *(_QWORD *)(v7 + 400);
          v13 = *(_QWORD *)(v7 + 416);
          v14 = (_QWORD *)(v11 + 8 * v12);
          v15 = (_QWORD *)(v11 + 8 * v9);
          if ( v14 != v15 )
          {
            do
              *v15++ = v13;
            while ( v14 != v15 );
            v11 = *(_QWORD *)(v7 + 400);
          }
          *(_DWORD *)(v7 + 408) = v10;
          goto LABEL_12;
        }
      }
      else
      {
        *(_DWORD *)(v7 + 408) = v10;
      }
    }
    v11 = *(_QWORD *)(v7 + 400);
LABEL_12:
    v18 = v8;
    v20 = (_QWORD *)v7;
    *(_QWORD *)(v11 + 8 * v6) = sub_1DBA290(v4 | 0x80000000);
    sub_1DBB110(v20, *(_QWORD *)(v20[50] + 8 * v6));
    result = (double *)v18(a1);
LABEL_5:
    if ( ++v4 == v3 )
      goto LABEL_16;
    goto LABEL_6;
  }
LABEL_16:
  if ( *(_QWORD *)v22 )
    return (double *)sub_16D7950(*(__int64 *)v22);
  return result;
}
