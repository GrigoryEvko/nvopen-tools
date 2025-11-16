// Function: sub_E4D630
// Address: 0xe4d630
//
__int64 __fastcall sub_E4D630(__int64 *a1)
{
  size_t v2; // rbx
  char *v3; // r15
  _BYTE *v4; // rax
  size_t v5; // r13
  __int64 v6; // rax
  __int64 v7; // r12
  unsigned __int8 *v8; // r8
  size_t v9; // rdx
  unsigned __int64 v10; // rax
  _BYTE *v11; // rdi
  unsigned __int64 v12; // rax
  _BYTE *v13; // rdi
  __int64 result; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  _BYTE *v17; // rdx
  size_t v18; // [rsp+0h] [rbp-40h]
  unsigned __int64 v19; // [rsp+8h] [rbp-38h]

  v2 = a1[62];
  if ( !v2 && a1[82] == a1[84] )
    return sub_A51310(a1[38], 0xAu);
  v3 = (char *)a1[61];
  do
  {
    sub_C66A60(a1[38], *(_DWORD *)(a1[39] + 396));
    if ( v2 )
    {
      v4 = memchr(v3, 10, v2);
      if ( v4 )
      {
        v5 = v4 - v3;
        v19 = v4 - v3 + 1;
        if ( v2 <= v4 - v3 )
          v5 = v2;
      }
      else
      {
        v19 = 0;
        v5 = v2;
      }
    }
    else
    {
      v19 = 0;
      v5 = 0;
    }
    v6 = a1[39];
    v7 = a1[38];
    v8 = *(unsigned __int8 **)(v6 + 48);
    v9 = *(_QWORD *)(v6 + 56);
    v10 = *(_QWORD *)(v7 + 24);
    v11 = *(_BYTE **)(v7 + 32);
    if ( v9 > v10 - (unsigned __int64)v11 )
    {
      v16 = sub_CB6200(a1[38], v8, v9);
      v11 = *(_BYTE **)(v16 + 32);
      v7 = v16;
      v10 = *(_QWORD *)(v16 + 24);
    }
    else if ( v9 )
    {
      v18 = v9;
      memcpy(v11, v8, v9);
      v17 = (_BYTE *)(*(_QWORD *)(v7 + 32) + v18);
      *(_QWORD *)(v7 + 32) = v17;
      v10 = *(_QWORD *)(v7 + 24);
      v11 = v17;
    }
    if ( v10 <= (unsigned __int64)v11 )
    {
      v7 = sub_CB5D20(v7, 32);
    }
    else
    {
      *(_QWORD *)(v7 + 32) = v11 + 1;
      *v11 = 32;
    }
    v12 = *(_QWORD *)(v7 + 24);
    v13 = *(_BYTE **)(v7 + 32);
    if ( v12 - (unsigned __int64)v13 < v5 )
    {
      v15 = sub_CB6200(v7, (unsigned __int8 *)v3, v5);
      v13 = *(_BYTE **)(v15 + 32);
      v7 = v15;
      v12 = *(_QWORD *)(v15 + 24);
    }
    else if ( v5 )
    {
      memcpy(v13, v3, v5);
      v12 = *(_QWORD *)(v7 + 24);
      v13 = (_BYTE *)(v5 + *(_QWORD *)(v7 + 32));
      *(_QWORD *)(v7 + 32) = v13;
    }
    if ( v12 <= (unsigned __int64)v13 )
    {
      sub_CB5D20(v7, 10);
    }
    else
    {
      *(_QWORD *)(v7 + 32) = v13 + 1;
      *v13 = 10;
    }
    result = v19;
    if ( v2 < v19 )
      break;
    v2 -= v19;
    v3 += v19;
  }
  while ( v2 );
  a1[62] = 0;
  return result;
}
