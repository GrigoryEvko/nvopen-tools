// Function: sub_386F7C0
// Address: 0x386f7c0
//
__int64 __fastcall sub_386F7C0(__int64 *a1, __int64 *a2, __int64 *a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r15
  __int64 *v7; // r12
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 result; // rax
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rdi
  __int64 v26; // [rsp+18h] [rbp-48h]
  __int64 v27; // [rsp+20h] [rbp-40h]
  bool v28; // [rsp+2Fh] [rbp-31h]

  v6 = a3;
  v7 = a1;
  if ( a3 != a4 && a1 != a2 )
  {
    do
    {
      v10 = v7[1];
      v11 = *v6;
      v26 = *v7;
      v27 = v6[1];
      v28 = *(_BYTE *)(sub_1456040(v27) + 8) == 15;
      if ( v28 == (*(_BYTE *)(sub_1456040(v10) + 8) == 15) )
      {
        if ( v26 == v11 )
        {
          if ( sub_1456260(v27) )
          {
            sub_1456260(v10);
          }
          else if ( sub_1456260(v10) )
          {
LABEL_5:
            v9 = *v6;
            a5 += 16;
            v6 += 2;
            *(_QWORD *)(a5 - 16) = v9;
            *(_QWORD *)(a5 - 8) = *(v6 - 1);
            if ( v7 == a2 )
              break;
            continue;
          }
        }
        else if ( v11 != sub_386EC30(v11, v26, a6) )
        {
          goto LABEL_5;
        }
      }
      else if ( *(_BYTE *)(sub_1456040(v27) + 8) == 15 )
      {
        goto LABEL_5;
      }
      v12 = *v7;
      a5 += 16;
      v7 += 2;
      *(_QWORD *)(a5 - 16) = v12;
      *(_QWORD *)(a5 - 8) = *(v7 - 1);
      if ( v7 == a2 )
        break;
    }
    while ( v6 != a4 );
  }
  v13 = (char *)a2 - (char *)v7;
  v14 = ((char *)a2 - (char *)v7) >> 4;
  if ( (char *)a2 - (char *)v7 <= 0 )
  {
    result = a5;
  }
  else
  {
    v15 = a5;
    do
    {
      v16 = *v7;
      v15 += 16;
      v7 += 2;
      *(_QWORD *)(v15 - 16) = v16;
      *(_QWORD *)(v15 - 8) = *(v7 - 1);
      --v14;
    }
    while ( v14 );
    v17 = 16;
    if ( v13 > 0 )
      v17 = v13;
    result = a5 + v17;
  }
  v19 = (char *)a4 - (char *)v6;
  v20 = ((char *)a4 - (char *)v6) >> 4;
  if ( (char *)a4 - (char *)v6 > 0 )
  {
    v21 = result;
    do
    {
      v22 = *v6;
      v21 += 16;
      v6 += 2;
      *(_QWORD *)(v21 - 16) = v22;
      *(_QWORD *)(v21 - 8) = *(v6 - 1);
      --v20;
    }
    while ( v20 );
    if ( v19 <= 0 )
      v19 = 16;
    result += v19;
  }
  return result;
}
