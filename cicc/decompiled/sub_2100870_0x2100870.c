// Function: sub_2100870
// Address: 0x2100870
//
__int64 __fastcall sub_2100870(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        int a5,
        double a6,
        double a7,
        double a8,
        double a9)
{
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 *v11; // rdx
  int v12; // eax
  int v13; // ebx
  int v15; // ebx
  __int64 v16; // r12
  double v17; // xmm4_8
  __int64 v18; // rcx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  int v21; // r10d
  unsigned int v22; // eax
  __int64 v23; // r9
  __int64 v24; // r11
  unsigned int v25; // eax
  __int64 v26; // rcx
  __int64 v27; // r12
  __int64 v28; // rdi
  __int64 v30; // rdi
  _QWORD *v31; // rsi
  _QWORD *v32; // rdx
  __int64 v33; // [rsp+0h] [rbp-A0h]
  unsigned int v34; // [rsp+10h] [rbp-90h]
  int v35; // [rsp+14h] [rbp-8Ch]
  __int64 v36; // [rsp+18h] [rbp-88h]
  int v37; // [rsp+18h] [rbp-88h]
  _QWORD v38[6]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v39; // [rsp+50h] [rbp-50h]
  __int64 v40; // [rsp+58h] [rbp-48h]
  int v41; // [rsp+60h] [rbp-40h]
  float (__fastcall *v42)(int, float); // [rsp+68h] [rbp-38h]

  v9 = *(_QWORD *)(a1 + 40);
  v10 = *(_QWORD *)(a1 + 32);
  v38[3] = a3;
  v11 = *(__int64 **)(a1 + 16);
  v38[4] = a4;
  v38[2] = v9;
  v42 = sub_20FF780;
  v12 = *(_DWORD *)(a1 + 64);
  v38[1] = v10;
  v38[5] = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v13 = *((_DWORD *)v11 + 2);
  v38[0] = a2;
  v35 = v13 - v12;
  if ( v13 == v12 )
  {
    v28 = 0;
    return j___libc_free_0(v28);
  }
  v15 = 0;
  while ( 1 )
  {
    v18 = (unsigned int)(v15 + v12);
    v19 = *v11;
    v20 = *(unsigned int *)(v10 + 408);
    v21 = *(_DWORD *)(v19 + 4 * v18);
    v22 = v21 & 0x7FFFFFFF;
    v23 = v21 & 0x7FFFFFFF;
    v24 = 8 * v23;
    if ( (v21 & 0x7FFFFFFFu) >= (unsigned int)v20 || (v16 = *(_QWORD *)(*(_QWORD *)(v10 + 400) + 8LL * v22)) == 0 )
    {
      v25 = v22 + 1;
      if ( (unsigned int)v20 < v25 )
      {
        v27 = v25;
        if ( v25 >= v20 )
        {
          if ( v25 > v20 )
          {
            if ( v25 > (unsigned __int64)*(unsigned int *)(v10 + 412) )
            {
              v33 = v21 & 0x7FFFFFFF;
              v34 = v25;
              v37 = v21;
              sub_16CD150(v10 + 400, (const void *)(v10 + 416), v25, 8, a5, v23);
              v20 = *(unsigned int *)(v10 + 408);
              v23 = v33;
              v24 = 8 * v33;
              v25 = v34;
              v21 = v37;
            }
            v26 = *(_QWORD *)(v10 + 400);
            v30 = *(_QWORD *)(v10 + 416);
            v31 = (_QWORD *)(v26 + 8 * v27);
            v32 = (_QWORD *)(v26 + 8 * v20);
            if ( v31 != v32 )
            {
              do
                *v32++ = v30;
              while ( v31 != v32 );
              v26 = *(_QWORD *)(v10 + 400);
            }
            *(_DWORD *)(v10 + 408) = v25;
            goto LABEL_9;
          }
        }
        else
        {
          *(_DWORD *)(v10 + 408) = v25;
        }
      }
      v26 = *(_QWORD *)(v10 + 400);
LABEL_9:
      v36 = v23;
      *(_QWORD *)(v26 + v24) = sub_1DBA290(v21);
      v16 = *(_QWORD *)(*(_QWORD *)(v10 + 400) + 8 * v36);
      sub_1DBB110((_QWORD *)v10, v16);
    }
    ++v15;
    sub_1E697B0(*(_QWORD **)(a1 + 24), *(_DWORD *)(v16 + 112));
    sub_20E2DF0(v38, v16, a6, a7, a8, a9, v17);
    if ( v15 == v35 )
      break;
    v10 = *(_QWORD *)(a1 + 32);
    v12 = *(_DWORD *)(a1 + 64);
    v11 = *(__int64 **)(a1 + 16);
  }
  v28 = v39;
  return j___libc_free_0(v28);
}
