// Function: sub_C0BDD0
// Address: 0xc0bdd0
//
void __fastcall sub_C0BDD0(__int64 a1, unsigned __int64 a2, int a3)
{
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // r8
  __int64 v6; // r9
  unsigned __int64 v7; // rcx
  int v8; // r12d
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // rcx
  __int64 v11; // r15
  __int64 *v12; // rax
  __int64 v13; // rsi
  __int64 *v14; // rdi
  __int64 v15; // rdx
  unsigned __int64 v16; // rsi
  int v17; // eax
  __int64 *v18; // rax
  __int64 v19; // rsi
  __int64 v21; // [rsp+8h] [rbp-48h]
  unsigned __int64 v22; // [rsp+10h] [rbp-40h]
  unsigned int v24; // [rsp+1Ch] [rbp-34h]

  if ( a2 > 1 )
  {
    v4 = a2;
    v5 = a3;
    v6 = ~(__int64)a3;
    while ( 1 )
    {
      v7 = *(unsigned int *)(*(_QWORD *)a1 + 8LL);
      if ( v7 <= v5 )
        v8 = -1;
      else
        v8 = *(unsigned __int8 *)(v7 + **(_QWORD **)a1 + v6);
      v9 = v4;
      v10 = 1;
      v11 = 0;
      do
      {
        while ( 1 )
        {
          v14 = (__int64 *)(a1 + 8 * v10);
          v15 = *v14;
          v16 = *(unsigned int *)(*v14 + 8);
          if ( v16 <= v5 )
            break;
          v17 = *(unsigned __int8 *)(v16 + *(_QWORD *)*v14 + v6);
          if ( v8 >= v17 )
            goto LABEL_10;
          v12 = (__int64 *)(a1 + 8 * v11);
          ++v10;
          ++v11;
          v13 = *v12;
          *v12 = v15;
          *v14 = v13;
LABEL_7:
          if ( v10 >= v9 )
            goto LABEL_12;
        }
        v17 = -1;
LABEL_10:
        if ( v8 <= v17 )
        {
          ++v10;
          goto LABEL_7;
        }
        --v9;
        v18 = (__int64 *)(a1 + 8 * v9);
        v19 = *v18;
        *v18 = v15;
        *v14 = v19;
      }
      while ( v10 < v9 );
LABEL_12:
      v24 = v5 + a3 - a3;
      v21 = v6;
      v22 = v5;
      sub_C0BDD0(a1, v11, v24);
      sub_C0BDD0(a1 + 8 * v9, v4 - v9, v24);
      if ( v8 != -1 )
      {
        a1 += 8 * v11;
        v4 = v9 - v11;
        v5 = v22 + 1;
        v6 = v21 - 1;
        if ( v9 - v11 > 1 )
          continue;
      }
      return;
    }
  }
}
