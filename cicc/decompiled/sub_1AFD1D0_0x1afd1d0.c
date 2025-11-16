// Function: sub_1AFD1D0
// Address: 0x1afd1d0
//
unsigned __int64 __fastcall sub_1AFD1D0(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v4; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  char v8; // r15
  unsigned __int8 *v9; // rcx
  __int64 v10; // rdi
  unsigned int v11; // esi
  _QWORD *v12; // rcx
  __int64 v13; // r9
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rcx
  unsigned __int64 v18; // rdx
  __int64 v19; // rdx
  _QWORD *v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rdi
  __int64 v23; // r8
  __int64 v24; // r9
  unsigned __int64 *v25; // r8
  unsigned int v26; // esi
  _QWORD *v27; // rdx
  __int64 v28; // r12
  int v29; // ecx
  int v30; // r10d
  int v31; // edx
  int v32; // r13d
  __int64 *v33; // [rsp-48h] [rbp-48h]
  __int64 v34; // [rsp-40h] [rbp-40h]

  result = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 0 )
  {
    v4 = 0;
    v34 = 24LL * (unsigned int)result;
    do
    {
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v6 = *(_QWORD *)(a1 - 8);
      else
        v6 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v7 = *(_QWORD *)(v6 + v4);
      v8 = 0;
      if ( *(_BYTE *)(v7 + 16) == 19 )
      {
        v9 = *(unsigned __int8 **)(v7 + 24);
        if ( (unsigned int)*v9 - 1 <= 1 )
        {
          v7 = *((_QWORD *)v9 + 17);
          v8 = 1;
        }
      }
      result = *(unsigned int *)(a2 + 24);
      if ( (_DWORD)result )
      {
        v10 = *(_QWORD *)(a2 + 8);
        v11 = (result - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v12 = (_QWORD *)(v10 + ((unsigned __int64)v11 << 6));
        v13 = v12[3];
        if ( v13 == v7 )
        {
LABEL_9:
          result = v10 + (result << 6);
          if ( v12 != (_QWORD *)result )
          {
            v14 = v12[7];
            v15 = sub_16498A0(a1);
            if ( v8 )
            {
              v33 = (__int64 *)v15;
              v20 = sub_1624210(v14);
              v14 = sub_1628DA0(v33, (__int64)v20);
            }
            if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
              v16 = *(_QWORD *)(a1 - 8);
            else
              v16 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
            result = v4 + v16;
            if ( *(_QWORD *)result )
            {
              v17 = *(_QWORD *)(result + 8);
              v18 = *(_QWORD *)(result + 16) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v18 = v17;
              if ( v17 )
                *(_QWORD *)(v17 + 16) = *(_QWORD *)(v17 + 16) & 3LL | v18;
            }
            *(_QWORD *)result = v14;
            if ( v14 )
            {
              v19 = *(_QWORD *)(v14 + 8);
              *(_QWORD *)(result + 8) = v19;
              if ( v19 )
                *(_QWORD *)(v19 + 16) = (result + 8) | *(_QWORD *)(v19 + 16) & 3LL;
              *(_QWORD *)(result + 16) = (v14 + 8) | *(_QWORD *)(result + 16) & 3LL;
              *(_QWORD *)(v14 + 8) = result;
            }
          }
        }
        else
        {
          v29 = 1;
          while ( v13 != -8 )
          {
            v30 = v29 + 1;
            v11 = (result - 1) & (v29 + v11);
            v12 = (_QWORD *)(v10 + ((unsigned __int64)v11 << 6));
            v13 = v12[3];
            if ( v13 == v7 )
              goto LABEL_9;
            v29 = v30;
          }
        }
      }
      v4 += 24;
    }
    while ( v4 != v34 );
    if ( *(_BYTE *)(a1 + 16) == 77 && (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 0 )
    {
      v21 = 0;
      v22 = 8LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      do
      {
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
          v23 = *(_QWORD *)(a1 - 8);
        else
          v23 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        result = *(unsigned int *)(a2 + 24);
        if ( (_DWORD)result )
        {
          v24 = *(_QWORD *)(a2 + 8);
          v25 = (unsigned __int64 *)(v21 + 24LL * *(unsigned int *)(a1 + 56) + 8 + v23);
          v26 = (result - 1) & (((unsigned int)*v25 >> 9) ^ ((unsigned int)*v25 >> 4));
          v27 = (_QWORD *)(v24 + ((unsigned __int64)v26 << 6));
          v28 = v27[3];
          if ( *v25 == v28 )
          {
LABEL_31:
            result = v24 + (result << 6);
            if ( v27 != (_QWORD *)result )
            {
              result = v27[7];
              *v25 = result;
            }
          }
          else
          {
            v31 = 1;
            while ( v28 != -8 )
            {
              v32 = v31 + 1;
              v26 = (result - 1) & (v31 + v26);
              v27 = (_QWORD *)(v24 + ((unsigned __int64)v26 << 6));
              v28 = v27[3];
              if ( *v25 == v28 )
                goto LABEL_31;
              v31 = v32;
            }
          }
        }
        v21 += 8;
      }
      while ( v21 != v22 );
    }
  }
  return result;
}
