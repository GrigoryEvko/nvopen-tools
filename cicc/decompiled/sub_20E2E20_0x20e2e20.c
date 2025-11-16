// Function: sub_20E2E20
// Address: 0x20e2e20
//
__int64 __fastcall sub_20E2E20(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11)
{
  __int64 v11; // r12
  int v12; // eax
  int v14; // ebx
  __int64 v15; // rsi
  unsigned int v16; // edx
  __int64 v17; // r8
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // r15
  unsigned int v21; // eax
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v25; // rdx
  _QWORD *v26; // rdi
  _QWORD *v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // [rsp+10h] [rbp-90h]
  __int64 v30; // [rsp+10h] [rbp-90h]
  __int64 v31; // [rsp+10h] [rbp-90h]
  unsigned int v32; // [rsp+18h] [rbp-88h]
  int v33; // [rsp+1Ch] [rbp-84h]
  _QWORD v34[6]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v35; // [rsp+50h] [rbp-50h]
  __int64 v36; // [rsp+58h] [rbp-48h]
  int v37; // [rsp+60h] [rbp-40h]
  __int64 v38; // [rsp+68h] [rbp-38h]

  v11 = *(_QWORD *)(a2 + 40);
  v34[4] = a5;
  v34[5] = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = a6;
  v12 = *(_DWORD *)(v11 + 32);
  v34[0] = a2;
  v34[1] = a1;
  v34[2] = a3;
  v34[3] = a4;
  v33 = v12;
  if ( !v12 )
  {
    v23 = 0;
    return j___libc_free_0(v23);
  }
  v14 = 0;
  do
  {
    while ( 1 )
    {
      v16 = v14 & 0x7FFFFFFF;
      v17 = v14 & 0x7FFFFFFF;
      v18 = *(_QWORD *)(*(_QWORD *)(v11 + 24) + 16 * v17 + 8);
      if ( !v18 )
        goto LABEL_5;
      if ( (*(_BYTE *)(v18 + 4) & 8) != 0 )
        break;
LABEL_8:
      v19 = *(unsigned int *)(a1 + 408);
      v20 = 8 * v17;
      if ( v16 < (unsigned int)v19 )
      {
        v15 = *(_QWORD *)(*(_QWORD *)(a1 + 400) + 8 * v17);
        if ( v15 )
          goto LABEL_4;
      }
      v21 = v16 + 1;
      if ( (unsigned int)v19 < v16 + 1 )
      {
        v25 = v21;
        if ( v21 >= v19 )
        {
          if ( v21 > v19 )
          {
            if ( v21 > (unsigned __int64)*(unsigned int *)(a1 + 412) )
            {
              v32 = v21;
              v31 = v21;
              sub_16CD150(a1 + 400, (const void *)(a1 + 416), v21, 8, v17, a6);
              v19 = *(unsigned int *)(a1 + 408);
              v21 = v32;
              v17 = v14 & 0x7FFFFFFF;
              v25 = v31;
            }
            v22 = *(_QWORD *)(a1 + 400);
            v26 = (_QWORD *)(v22 + 8 * v25);
            v27 = (_QWORD *)(v22 + 8 * v19);
            v28 = *(_QWORD *)(a1 + 416);
            if ( v26 != v27 )
            {
              do
                *v27++ = v28;
              while ( v26 != v27 );
              v22 = *(_QWORD *)(a1 + 400);
            }
            *(_DWORD *)(a1 + 408) = v21;
            goto LABEL_11;
          }
        }
        else
        {
          *(_DWORD *)(a1 + 408) = v21;
        }
      }
      v22 = *(_QWORD *)(a1 + 400);
LABEL_11:
      v29 = v17;
      *(_QWORD *)(v22 + v20) = sub_1DBA290(v14 | 0x80000000);
      v30 = *(_QWORD *)(*(_QWORD *)(a1 + 400) + 8 * v29);
      sub_1DBB110((_QWORD *)a1, v30);
      v15 = v30;
LABEL_4:
      sub_20E2DF0(v34, v15, a7, a8, a9, a10, a11);
LABEL_5:
      if ( v33 == ++v14 )
        goto LABEL_15;
    }
    while ( 1 )
    {
      v18 = *(_QWORD *)(v18 + 32);
      if ( !v18 )
        break;
      if ( (*(_BYTE *)(v18 + 4) & 8) == 0 )
        goto LABEL_8;
    }
    ++v14;
  }
  while ( v33 != v14 );
LABEL_15:
  v23 = v35;
  return j___libc_free_0(v23);
}
