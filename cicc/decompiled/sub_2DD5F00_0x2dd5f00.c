// Function: sub_2DD5F00
// Address: 0x2dd5f00
//
void __fastcall sub_2DD5F00(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rbx
  char v6; // r15
  __int64 v7; // rax
  unsigned __int64 v8; // rbx
  char v9; // r15
  __int64 *v10; // r15
  __int64 v11; // r14
  char v12; // bl
  __int64 v13; // rax
  __int64 v14; // r14
  unsigned __int64 v15; // rbx
  __int64 *v16; // [rsp+10h] [rbp-60h]
  __int64 *v17; // [rsp+18h] [rbp-58h]
  __int64 v18; // [rsp+20h] [rbp-50h]
  __int64 v19; // [rsp+28h] [rbp-48h]
  __int64 v20; // [rsp+28h] [rbp-48h]
  __int64 v21; // [rsp+28h] [rbp-48h]

  if ( a1 != a2 && a2 != a1 + 1 )
  {
    v16 = a1 + 1;
    do
    {
      while ( 1 )
      {
        v5 = *(_QWORD *)(*v16 + 24);
        v19 = *a1;
        v6 = sub_AE5020(a3, v5);
        v7 = sub_9208B0(a3, v5);
        v20 = *(_QWORD *)(v19 + 24);
        v8 = ((1LL << v6) + ((unsigned __int64)(v7 + 7) >> 3) - 1) >> v6 << v6;
        v9 = sub_AE5020(a3, v20);
        if ( ((1LL << v9) + ((unsigned __int64)(sub_9208B0(a3, v20) + 7) >> 3) - 1) >> v9 << v9 > v8 )
          break;
        v10 = v16;
        v18 = *v16;
        while ( 1 )
        {
          v11 = *(v10 - 1);
          v17 = v10--;
          v21 = *(_QWORD *)(v18 + 24);
          v12 = sub_AE5020(a3, v21);
          v13 = sub_9208B0(a3, v21);
          v14 = *(_QWORD *)(v11 + 24);
          v15 = ((1LL << v12) + ((unsigned __int64)(v13 + 7) >> 3) - 1) >> v12 << v12;
          LOBYTE(v21) = sub_AE5020(a3, v14);
          if ( ((1LL << v21) + ((unsigned __int64)(sub_9208B0(a3, v14) + 7) >> 3) - 1) >> v21 << v21 <= v15 )
            break;
          v10[1] = *v10;
        }
        ++v16;
        *v17 = v18;
        if ( a2 == v16 )
          return;
      }
      v4 = *v16;
      if ( a1 != v16 )
        memmove(a1 + 1, a1, (char *)v16 - (char *)a1);
      ++v16;
      *a1 = v4;
    }
    while ( a2 != v16 );
  }
}
