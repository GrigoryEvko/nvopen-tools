// Function: sub_34F5750
// Address: 0x34f5750
//
__int64 __fastcall sub_34F5750(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rbx
  __int64 result; // rax
  _BYTE *v4; // r12
  _BYTE *v6; // r15
  __int64 v7; // r8
  _BYTE *v8; // rbx
  unsigned __int64 v9; // rdx
  __int64 v10; // rbx
  unsigned int v11; // eax
  __int64 v12; // rcx
  __int64 *v13; // rbx
  __int64 v14; // rax
  __int64 v15; // r9
  unsigned __int64 v16; // r10
  __int64 *v17; // rax
  __int64 *v18; // rsi
  unsigned __int64 v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h]
  int v21; // [rsp+1Ch] [rbp-34h]

  v2 = *(_BYTE **)(a1 + 32);
  result = 5LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF);
  v4 = &v2[40 * (*(_DWORD *)(a1 + 40) & 0xFFFFFF)];
  if ( v2 != v4 )
  {
    while ( 1 )
    {
      v6 = v2;
      result = sub_2DADC00(v2);
      if ( (_BYTE)result )
        break;
      v2 += 40;
      if ( v4 == v2 )
        return result;
    }
    if ( v4 != v2 )
    {
      while ( 1 )
      {
        v7 = *((unsigned int *)v6 + 2);
        if ( (int)v7 < 0 )
        {
          v9 = *(unsigned int *)(a2 + 160);
          result = v7 & 0x7FFFFFFF;
          v10 = 8LL * (unsigned int)result;
          if ( (unsigned int)result >= (unsigned int)v9
            || !*(_QWORD *)(*(_QWORD *)(a2 + 152) + 8LL * (unsigned int)result) )
          {
            break;
          }
        }
LABEL_7:
        v8 = v6 + 40;
        if ( v6 + 40 != v4 )
        {
          while ( 1 )
          {
            v6 = v8;
            result = sub_2DADC00(v8);
            if ( (_BYTE)result )
              break;
            v8 += 40;
            if ( v4 == v8 )
              return result;
          }
          if ( v4 != v8 )
            continue;
        }
        return result;
      }
      v11 = result + 1;
      if ( (unsigned int)v9 < v11 && v11 != v9 )
      {
        if ( v11 >= v9 )
        {
          v15 = *(_QWORD *)(a2 + 168);
          v16 = v11 - v9;
          if ( v11 > (unsigned __int64)*(unsigned int *)(a2 + 164) )
          {
            v19 = v11 - v9;
            v20 = *(_QWORD *)(a2 + 168);
            v21 = *((_DWORD *)v6 + 2);
            sub_C8D5F0(a2 + 152, (const void *)(a2 + 168), v11, 8u, v7, v15);
            v9 = *(unsigned int *)(a2 + 160);
            v16 = v19;
            v15 = v20;
            LODWORD(v7) = v21;
          }
          v12 = *(_QWORD *)(a2 + 152);
          v17 = (__int64 *)(v12 + 8 * v9);
          v18 = &v17[v16];
          if ( v17 != v18 )
          {
            do
              *v17++ = v15;
            while ( v18 != v17 );
            LODWORD(v9) = *(_DWORD *)(a2 + 160);
            v12 = *(_QWORD *)(a2 + 152);
          }
          *(_DWORD *)(a2 + 160) = v16 + v9;
          goto LABEL_17;
        }
        *(_DWORD *)(a2 + 160) = v11;
      }
      v12 = *(_QWORD *)(a2 + 152);
LABEL_17:
      v13 = (__int64 *)(v12 + v10);
      v14 = sub_2E10F30(v7);
      *v13 = v14;
      result = sub_2E11E80((_QWORD *)a2, v14);
      goto LABEL_7;
    }
  }
  return result;
}
