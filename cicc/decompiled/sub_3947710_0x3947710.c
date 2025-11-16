// Function: sub_3947710
// Address: 0x3947710
//
__int64 __fastcall sub_3947710(__int64 a1, unsigned __int8 *a2, size_t a3)
{
  __int64 *v4; // rbx
  __int64 v6; // r12
  void *v7; // rcx
  void *v8; // r15
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // r14
  __int64 *v12; // rcx
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned int v16; // [rsp+1Ch] [rbp-34h]

  v16 = sub_16D19C0(a1, a2, a3);
  v4 = (__int64 *)(*(_QWORD *)a1 + 8LL * v16);
  if ( *v4 )
  {
    if ( *v4 != -8 )
      return *(_QWORD *)a1 + 8LL * v16;
    --*(_DWORD *)(a1 + 16);
  }
  v6 = malloc(a3 + 153);
  if ( !v6 )
  {
    if ( a3 == -153 )
    {
      v15 = malloc(1u);
      if ( v15 )
      {
        v7 = (void *)(v15 + 152);
        v6 = v15;
        goto LABEL_20;
      }
    }
    sub_16BD1C0("Allocation failed", 1u);
  }
  v7 = (void *)(v6 + 152);
  if ( a3 + 1 > 1 )
LABEL_20:
    v7 = memcpy(v7, a2, a3);
  *((_BYTE *)v7 + a3) = 0;
  *(_QWORD *)v6 = a3;
  v8 = (void *)(v6 + 120);
  memset((void *)(v6 + 8), 0, 0x90u);
  *(_QWORD *)(v6 + 72) = v6 + 120;
  *(_DWORD *)(v6 + 28) = 16;
  *(_QWORD *)(v6 + 80) = 1;
  *(_DWORD *)(v6 + 104) = 1065353216;
  v9 = sub_222D860(v6 + 104, 0x100u);
  v11 = v9;
  if ( v9 > *(_QWORD *)(v6 + 80) )
  {
    if ( v9 == 1 )
    {
      *(_QWORD *)(v6 + 120) = 0;
    }
    else
    {
      if ( v9 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(v6 + 104, 256, v10);
      v8 = (void *)sub_22077B0(8 * v9);
      memset(v8, 0, 8 * v11);
    }
    *(_QWORD *)(v6 + 72) = v8;
    *(_QWORD *)(v6 + 80) = v11;
  }
  *(_QWORD *)(v6 + 128) = 0;
  *(_QWORD *)(v6 + 136) = 0;
  *(_QWORD *)(v6 + 144) = 0;
  *v4 = v6;
  ++*(_DWORD *)(a1 + 12);
  v12 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v16));
  if ( *v12 == -8 || !*v12 )
  {
    v13 = v12 + 1;
    do
    {
      do
      {
        v14 = *v13;
        v12 = v13++;
      }
      while ( !v14 );
    }
    while ( v14 == -8 );
  }
  return (__int64)v12;
}
