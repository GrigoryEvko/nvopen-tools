// Function: sub_1CB7C40
// Address: 0x1cb7c40
//
__int64 __fastcall sub_1CB7C40(__int64 a1, unsigned __int64 a2)
{
  unsigned int v3; // eax
  unsigned int v4; // r13d
  unsigned __int64 v5; // rsi
  int v6; // eax
  int v7; // edx
  int v9; // r15d
  unsigned int v10; // r15d
  int v11; // r9d
  unsigned __int64 v12; // r10
  unsigned int v13; // ebx
  void *v14; // rax
  _QWORD *v15; // rax
  _QWORD *v16; // r11
  _QWORD *v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // rax
  unsigned int v22; // ebx
  unsigned int v23; // eax
  int v24; // edx
  _QWORD *v25; // r11
  int v26; // r9d
  int v27; // eax
  unsigned __int64 v28; // [rsp+8h] [rbp-58h]
  int v29; // [rsp+14h] [rbp-4Ch]
  unsigned __int64 v30; // [rsp+18h] [rbp-48h]
  int v31; // [rsp+20h] [rbp-40h]
  _QWORD *v32; // [rsp+20h] [rbp-40h]
  _QWORD *v33; // [rsp+20h] [rbp-40h]
  __int64 n; // [rsp+28h] [rbp-38h]

  LOBYTE(v3) = sub_1594510(a2);
  v4 = v3;
  if ( (_BYTE)v3 )
  {
    v5 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) == 15 )
    {
      v9 = sub_1CB76C0((unsigned int *)a1, v5);
      if ( v9 != (unsigned int)sub_1CB76C0((unsigned int *)a1, a2) )
      {
        v7 = v9;
        goto LABEL_4;
      }
    }
    else
    {
      v6 = sub_1CB76C0((unsigned int *)a1, a2);
      v7 = *(_DWORD *)(a1 + 4);
      if ( v7 != v6 )
      {
LABEL_4:
        sub_1CB7560((_QWORD *)a1, a2, v7);
        return v4;
      }
    }
    return 0;
  }
  v10 = sub_1594530(a2);
  if ( !(_BYTE)v10 )
    return 0;
  v11 = sub_1CB76C0((unsigned int *)a1, a2);
  v12 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) == 15 )
  {
    v13 = (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) - 1;
    if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) == 1 )
    {
      n = 0;
      v16 = 0;
    }
    else
    {
      v30 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v31 = v11;
      n = 8LL * v13;
      v14 = (void *)sub_22077B0(n);
      v15 = memset(v14, 0, n);
      v11 = v31;
      v12 = v30;
      v16 = v15;
    }
    v17 = v16;
    LODWORD(v18) = 0;
    while ( (_DWORD)v18 != v13 )
    {
      ++v17;
      v18 = (unsigned int)(v18 + 1);
      v19 = *(_QWORD *)(a2 + 24 * (v18 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      *(v17 - 1) = v19;
      if ( *(_BYTE *)(v19 + 16) != 13 )
      {
        v33 = v16;
        v24 = *(_DWORD *)(a1 + 4);
        goto LABEL_19;
      }
    }
    v29 = v11;
    v28 = v12;
    v20 = *(_QWORD *)(a1 + 280);
    v32 = v16;
    v21 = sub_16348C0(a2);
    v22 = sub_15A9FF0(v20, v21, v32, n >> 3);
    v23 = sub_1CB76C0((unsigned int *)a1, v28);
    v24 = *(_DWORD *)a1;
    v25 = v32;
    v26 = v29;
    if ( v23 != *(_DWORD *)a1 )
    {
      v27 = sub_1CB71C0(a1, v22, v23);
      v25 = v32;
      v26 = v29;
      v24 = v27;
    }
    if ( v24 != v26 )
    {
      v33 = v25;
LABEL_19:
      v4 = v10;
      sub_1CB7560((_QWORD *)a1, a2, v24);
      v25 = v33;
    }
    if ( v25 )
      j_j___libc_free_0(v25, n);
  }
  return v4;
}
