// Function: sub_E20AE0
// Address: 0xe20ae0
//
unsigned __int64 __fastcall sub_E20AE0(__int64 **a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // r12
  __int64 *v4; // r14
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rcx
  _QWORD *v10; // rdx
  _QWORD *v11; // rax
  __int64 *v13; // rax
  __int64 *v14; // r14
  __int64 *v15; // rdx
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 *v18; // rax
  unsigned __int64 *v19; // rax
  unsigned __int64 *v20; // r14
  unsigned __int64 *v21; // rdx

  v2 = **a1;
  v3 = (v2 + (*a1)[1] + 7) & 0xFFFFFFFFFFFFFFF8LL;
  (*a1)[1] = v3 - v2 + 24;
  v4 = *a1;
  v5 = (*a1)[1];
  if ( v5 > (*a1)[2] )
  {
    v16 = (__int64 *)sub_22077B0(32);
    v4 = v16;
    if ( v16 )
    {
      *v16 = 0;
      v16[1] = 0;
      v16[2] = 0;
      v16[3] = 0;
    }
    v17 = sub_2207820(4096);
    v4[2] = 4096;
    *v4 = v17;
    v3 = v17;
    v18 = *a1;
    v4[1] = 24;
    v4[3] = (__int64)v18;
    *a1 = v4;
    if ( !v3 )
    {
      v4[1] = 56;
      v7 = 24;
      goto LABEL_6;
    }
    v6 = v3;
    *(_DWORD *)(v3 + 8) = 20;
    *(_QWORD *)(v3 + 16) = 0;
    *(_QWORD *)v3 = &unk_49E1240;
    v5 = 24;
  }
  else if ( v3 )
  {
    *(_DWORD *)(v3 + 8) = 20;
    *(_QWORD *)(v3 + 16) = 0;
    *(_QWORD *)v3 = &unk_49E1240;
    v4 = *a1;
    v6 = **a1;
    v5 = (*a1)[1];
  }
  else
  {
    v6 = *v4;
  }
  v7 = (v6 + v5 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v4[1] = v7 - v6 + 32;
  if ( (*a1)[1] > (unsigned __int64)(*a1)[2] )
  {
    v19 = (unsigned __int64 *)sub_22077B0(32);
    v20 = v19;
    if ( v19 )
    {
      *v19 = 0;
      v19[1] = 0;
      v19[2] = 0;
      v19[3] = 0;
    }
    v7 = sub_2207820(4096);
    v21 = (unsigned __int64 *)*a1;
    v20[2] = 4096;
    *v20 = v7;
    v20[3] = (unsigned __int64)v21;
    *a1 = (__int64 *)v20;
    v20[1] = 32;
  }
  if ( !v7 )
  {
    *(_QWORD *)(v3 + 16) = 0;
    MEMORY[0x18] = 0;
    BUG();
  }
LABEL_6:
  *(_DWORD *)(v7 + 8) = 19;
  *(_QWORD *)(v7 + 16) = 0;
  *(_QWORD *)(v7 + 24) = 0;
  *(_QWORD *)v7 = &unk_49E0EC0;
  *(_QWORD *)(v3 + 16) = v7;
  *(_QWORD *)(v7 + 24) = 1;
  v8 = *(_QWORD *)(v3 + 16);
  v9 = **a1;
  v10 = (_QWORD *)((v9 + (*a1)[1] + 7) & 0xFFFFFFFFFFFFFFF8LL);
  (*a1)[1] = (__int64)v10 - v9 + 8;
  if ( (*a1)[1] > (unsigned __int64)(*a1)[2] )
  {
    v13 = (__int64 *)sub_22077B0(32);
    v14 = v13;
    if ( v13 )
    {
      *v13 = 0;
      v13[1] = 0;
      v13[2] = 0;
      v13[3] = 0;
    }
    v11 = (_QWORD *)sub_2207820(4096);
    v15 = *a1;
    v14[2] = 4096;
    *v14 = (__int64)v11;
    v14[3] = (__int64)v15;
    *a1 = v14;
    v14[1] = 8;
    if ( v11 )
      *v11 = 0;
  }
  else
  {
    v11 = 0;
    if ( v10 )
    {
      *v10 = 0;
      v11 = v10;
    }
  }
  *(_QWORD *)(v8 + 16) = v11;
  **(_QWORD **)(*(_QWORD *)(v3 + 16) + 16LL) = a2;
  return v3;
}
