// Function: sub_1EB3F20
// Address: 0x1eb3f20
//
__int64 __fastcall sub_1EB3F20(_QWORD *a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned int v6; // eax
  __int64 v7; // r15
  __int64 v8; // r8
  __int64 v10; // r13
  unsigned __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r12
  unsigned int v15; // r12d
  __int64 v16; // rcx
  __int64 v17; // rax
  _QWORD *v18; // rsi
  _QWORD *v19; // rax
  __int64 v20; // rdx

  v6 = a2 & 0x7FFFFFFF;
  v7 = a2 & 0x7FFFFFFF;
  v8 = 8 * v7;
  v10 = a1[33];
  v11 = *(unsigned int *)(v10 + 408);
  if ( (a2 & 0x7FFFFFFFu) < (unsigned int)v11 )
  {
    v12 = *(_QWORD *)(v10 + 400);
    v13 = *(_QWORD *)(v12 + 8LL * v6);
    if ( v13 )
      goto LABEL_3;
  }
  v15 = v6 + 1;
  if ( (unsigned int)v11 < v6 + 1 )
  {
    v17 = v15;
    if ( v15 < v11 )
    {
      *(_DWORD *)(v10 + 408) = v15;
    }
    else if ( v15 > v11 )
    {
      if ( v15 > (unsigned __int64)*(unsigned int *)(v10 + 412) )
      {
        sub_16CD150(v10 + 400, (const void *)(v10 + 416), v15, 8, 8 * a2, a6);
        v11 = *(unsigned int *)(v10 + 408);
        v8 = 8LL * (a2 & 0x7FFFFFFF);
        v17 = v15;
      }
      v16 = *(_QWORD *)(v10 + 400);
      v18 = (_QWORD *)(v16 + 8 * v17);
      v19 = (_QWORD *)(v16 + 8 * v11);
      v20 = *(_QWORD *)(v10 + 416);
      if ( v18 != v19 )
      {
        do
          *v19++ = v20;
        while ( v18 != v19 );
        v16 = *(_QWORD *)(v10 + 400);
      }
      *(_DWORD *)(v10 + 408) = v15;
      goto LABEL_7;
    }
  }
  v16 = *(_QWORD *)(v10 + 400);
LABEL_7:
  *(_QWORD *)(v16 + v8) = sub_1DBA290(a2);
  v13 = *(_QWORD *)(*(_QWORD *)(v10 + 400) + 8 * v7);
  sub_1DBB110((_QWORD *)v10, v13);
LABEL_3:
  if ( *(_DWORD *)(*(_QWORD *)(a1[32] + 264LL) + 4 * v7) )
  {
    sub_21031A0(a1[34], v13, v11, v12, v8);
    return 1;
  }
  else
  {
    *(_DWORD *)(v13 + 72) = 0;
    *(_DWORD *)(v13 + 8) = 0;
    return 0;
  }
}
