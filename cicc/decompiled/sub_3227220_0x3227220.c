// Function: sub_3227220
// Address: 0x3227220
//
__int64 __fastcall sub_3227220(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5, __int64 a6)
{
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r15
  __int64 v16; // rdx
  char *v17; // r12
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rsi
  int v20; // eax
  _QWORD *v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // rdx
  char *v26; // rcx
  unsigned __int64 v27; // rsi
  __int64 v28; // r9
  int v29; // eax
  _QWORD *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdi
  char *v33; // r12
  __int64 v34; // rdi
  char *v35; // r12
  __int64 v36; // [rsp+8h] [rbp-48h]
  _QWORD v37[7]; // [rsp+18h] [rbp-38h] BYREF

  sub_3223A50(a1, a2, (__int64)a4, *(_QWORD *)(a3 + 8));
  if ( *a4 == 26 )
  {
    v12 = sub_22077B0(0x60u);
    v15 = v12;
    if ( v12 )
    {
      *(_QWORD *)(v12 + 8) = a4;
      *(_QWORD *)(v12 + 16) = a5;
      *(_QWORD *)(v12 + 24) = 0;
      *(_DWORD *)(v12 + 32) = 0;
      *(_QWORD *)v12 = &unk_4A35790;
      *(_WORD *)(v12 + 88) = 0;
    }
    v16 = *(unsigned int *)(a1 + 768);
    v37[0] = v12;
    v17 = (char *)v37;
    v18 = *(_QWORD *)(a1 + 760);
    v19 = v16 + 1;
    v20 = v16;
    if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 772) )
    {
      v32 = a1 + 760;
      if ( v18 > (unsigned __int64)v37 || (unsigned __int64)v37 >= v18 + 8 * v16 )
      {
        sub_3227150(v32, v19, v16, v18, v13, v14);
        v16 = *(unsigned int *)(a1 + 768);
        v18 = *(_QWORD *)(a1 + 760);
        v20 = *(_DWORD *)(a1 + 768);
      }
      else
      {
        v33 = (char *)v37 - v18;
        sub_3227150(v32, v19, v16, v18, v13, v14);
        v18 = *(_QWORD *)(a1 + 760);
        v16 = *(unsigned int *)(a1 + 768);
        v17 = &v33[v18];
        v20 = *(_DWORD *)(a1 + 768);
      }
    }
    v21 = (_QWORD *)(v18 + 8 * v16);
    if ( v21 )
    {
      *v21 = *(_QWORD *)v17;
      *(_QWORD *)v17 = 0;
      v15 = v37[0];
      v20 = *(_DWORD *)(a1 + 768);
    }
    v22 = (unsigned int)(v20 + 1);
    *(_DWORD *)(a1 + 768) = v22;
    if ( v15 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v15 + 8LL))(v15);
      v22 = *(unsigned int *)(a1 + 768);
    }
    sub_3245B60(a1 + 3080, a3, *(_QWORD *)(*(_QWORD *)(a1 + 760) + 8 * v22 - 8));
  }
  else if ( *a4 == 27 )
  {
    v23 = sub_22077B0(0x30u);
    v24 = v23;
    if ( v23 )
    {
      *(_QWORD *)(v23 + 8) = a4;
      *(_QWORD *)(v23 + 16) = a5;
      *(_QWORD *)(v23 + 24) = 0;
      *(_DWORD *)(v23 + 32) = 1;
      *(_QWORD *)(v23 + 40) = a6;
      *(_QWORD *)v23 = &unk_4A357B0;
    }
    v25 = *(unsigned int *)(a1 + 768);
    v37[0] = v23;
    v26 = (char *)v37;
    v27 = *(_QWORD *)(a1 + 760);
    v28 = v25 + 1;
    v29 = v25;
    if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 772) )
    {
      v36 = v24;
      v34 = a1 + 760;
      if ( v27 > (unsigned __int64)v37 || (unsigned __int64)v37 >= v27 + 8 * v25 )
      {
        sub_3227150(v34, v25 + 1, v25, (__int64)v37, v24, v28);
        v25 = *(unsigned int *)(a1 + 768);
        v24 = v36;
        v26 = (char *)v37;
        v27 = *(_QWORD *)(a1 + 760);
        v29 = *(_DWORD *)(a1 + 768);
      }
      else
      {
        v35 = (char *)v37 - v27;
        sub_3227150(v34, v25 + 1, v25, (__int64)v37 - v27, v24, v28);
        v27 = *(_QWORD *)(a1 + 760);
        v25 = *(unsigned int *)(a1 + 768);
        v24 = v36;
        v26 = &v35[v27];
        v29 = *(_DWORD *)(a1 + 768);
      }
    }
    v30 = (_QWORD *)(v27 + 8 * v25);
    if ( v30 )
    {
      *v30 = *(_QWORD *)v26;
      *(_QWORD *)v26 = 0;
      v24 = v37[0];
      v29 = *(_DWORD *)(a1 + 768);
    }
    v31 = (unsigned int)(v29 + 1);
    *(_DWORD *)(a1 + 768) = v31;
    if ( v24 )
    {
      (*(void (__fastcall **)(__int64, unsigned __int64, _QWORD *, char *))(*(_QWORD *)v24 + 8LL))(v24, v27, v30, v26);
      v31 = *(unsigned int *)(a1 + 768);
    }
    sub_3246200(a1 + 3080, a3, *(_QWORD *)(*(_QWORD *)(a1 + 760) + 8 * v31 - 8), v26);
  }
  return *(_QWORD *)(*(_QWORD *)(a1 + 760) + 8LL * *(unsigned int *)(a1 + 768) - 8);
}
