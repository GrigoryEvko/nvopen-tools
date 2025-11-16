// Function: sub_2A6A450
// Address: 0x2a6a450
//
__int64 __fastcall sub_2A6A450(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // r12
  __int64 result; // rax
  int v5; // r13d
  unsigned int v6; // ebx
  unsigned int v7; // edx
  unsigned __int8 *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // r13
  __int64 v14; // rcx
  __int64 v15; // rdx
  int v16; // r10d
  __int64 *v17; // rdi
  unsigned int v18; // eax
  __int64 *v19; // rsi
  __int64 v20; // r9
  unsigned __int8 *v21; // rsi
  int v22; // eax
  int v23; // edx
  int v24; // esi
  unsigned __int8 *v25; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v26; // [rsp+18h] [rbp-28h] BYREF

  v3 = (unsigned __int8 *)a2;
  result = *(_QWORD *)(a2 + 8);
  v25 = (unsigned __int8 *)a2;
  if ( *(_BYTE *)(result + 8) != 15 )
  {
    v12 = *(unsigned int *)(a1 + 160);
    v13 = a1 + 136;
    if ( (_DWORD)v12 )
    {
      v14 = (unsigned int)(v12 - 1);
      v15 = *(_QWORD *)(a1 + 144);
      v16 = 1;
      v17 = 0;
      v18 = v14 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = (__int64 *)(v15 + 48LL * v18);
      v20 = *v19;
      if ( v3 == (unsigned __int8 *)*v19 )
      {
LABEL_9:
        v21 = (unsigned __int8 *)(v19 + 1);
        return sub_2A634B0(a1, v21, (__int64)v3, v14, v12, v20);
      }
      while ( v20 != -4096 )
      {
        if ( !v17 && v20 == -8192 )
          v17 = v19;
        v18 = v14 & (v16 + v18);
        v19 = (__int64 *)(v15 + 48LL * v18);
        v20 = *v19;
        if ( v3 == (unsigned __int8 *)*v19 )
          goto LABEL_9;
        ++v16;
      }
      v22 = *(_DWORD *)(a1 + 152);
      if ( !v17 )
        v17 = v19;
      ++*(_QWORD *)(a1 + 136);
      v23 = v22 + 1;
      v26 = v17;
      if ( 4 * (v22 + 1) < (unsigned int)(3 * v12) )
      {
        v14 = (__int64)v3;
        if ( (int)v12 - *(_DWORD *)(a1 + 156) - v23 > (unsigned int)v12 >> 3 )
        {
LABEL_21:
          *(_DWORD *)(a1 + 152) = v23;
          if ( *v17 != -4096 )
            --*(_DWORD *)(a1 + 156);
          *v17 = v14;
          v21 = (unsigned __int8 *)(v17 + 1);
          *((_WORD *)v17 + 4) = 0;
          return sub_2A634B0(a1, v21, (__int64)v3, v14, v12, v20);
        }
        v24 = v12;
LABEL_26:
        sub_2A68410(v13, v24);
        sub_2A65730(v13, (__int64 *)&v25, &v26);
        v14 = (__int64)v25;
        v17 = v26;
        v23 = *(_DWORD *)(a1 + 152) + 1;
        goto LABEL_21;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 136);
      v26 = 0;
    }
    v24 = 2 * v12;
    goto LABEL_26;
  }
  v5 = *(_DWORD *)(result + 12);
  v6 = 0;
  if ( v5 )
  {
    while ( 1 )
    {
      v7 = v6++;
      v8 = sub_2A6A1C0(a1, v3, v7);
      result = sub_2A634B0(a1, v8, (__int64)v3, v9, v10, v11);
      if ( v6 == v5 )
        break;
      v3 = v25;
    }
  }
  return result;
}
