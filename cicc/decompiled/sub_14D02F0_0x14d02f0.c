// Function: sub_14D02F0
// Address: 0x14d02f0
//
__int64 __fastcall sub_14D02F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // r13
  _QWORD *v6; // rbx
  _QWORD *v7; // r12
  _QWORD *v8; // rax
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 *v11; // rax
  __int64 *v12; // rdi
  unsigned int v13; // r8d
  __int64 *v14; // rsi
  _QWORD *v15; // rdx
  __int64 v18; // [rsp+10h] [rbp-40h]
  __int64 v19; // [rsp+18h] [rbp-38h]

  result = *(unsigned int *)(a2 + 8);
  v18 = 0;
  if ( (int)result > 0 )
  {
    while ( 1 )
    {
      v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 8 * v18) + 8LL);
      v19 = *(_QWORD *)(*(_QWORD *)a2 + 8 * v18);
      if ( v5 )
      {
        v6 = *(_QWORD **)(a3 + 16);
        do
        {
          v9 = sub_1648700(v5);
          v8 = *(_QWORD **)(a3 + 8);
          if ( v6 == v8 )
          {
            v10 = *(unsigned int *)(a3 + 28);
            v7 = &v6[v10];
            if ( v6 == v7 )
            {
              v15 = v6;
            }
            else
            {
              do
              {
                if ( v9 == *v8 )
                  break;
                ++v8;
              }
              while ( v7 != v8 );
              v15 = &v6[v10];
            }
LABEL_17:
            while ( v15 != v8 )
            {
              if ( *v8 < 0xFFFFFFFFFFFFFFFELL )
                goto LABEL_7;
              ++v8;
            }
            if ( v7 == v8 )
              goto LABEL_19;
          }
          else
          {
            v7 = &v6[*(unsigned int *)(a3 + 24)];
            v8 = (_QWORD *)sub_16CC9F0(a3, v9);
            if ( v9 == *v8 )
            {
              v6 = *(_QWORD **)(a3 + 16);
              if ( v6 == *(_QWORD **)(a3 + 8) )
                v15 = &v6[*(unsigned int *)(a3 + 28)];
              else
                v15 = &v6[*(unsigned int *)(a3 + 24)];
              goto LABEL_17;
            }
            v6 = *(_QWORD **)(a3 + 16);
            if ( v6 == *(_QWORD **)(a3 + 8) )
            {
              v8 = &v6[*(unsigned int *)(a3 + 28)];
              v15 = v8;
              goto LABEL_17;
            }
            v8 = &v6[*(unsigned int *)(a3 + 24)];
LABEL_7:
            if ( v7 == v8 )
              goto LABEL_19;
          }
          v5 = *(_QWORD *)(v5 + 8);
        }
        while ( v5 );
      }
      v11 = *(__int64 **)(a3 + 8);
      if ( *(__int64 **)(a3 + 16) != v11 )
        break;
      v12 = &v11[*(unsigned int *)(a3 + 28)];
      v13 = *(_DWORD *)(a3 + 28);
      if ( v11 == v12 )
      {
LABEL_35:
        if ( v13 >= *(_DWORD *)(a3 + 24) )
          break;
        *(_DWORD *)(a3 + 28) = v13 + 1;
        *v12 = v19;
        ++*(_QWORD *)a3;
      }
      else
      {
        v14 = 0;
        while ( v19 != *v11 )
        {
          if ( *v11 == -2 )
            v14 = v11;
          if ( v12 == ++v11 )
          {
            if ( !v14 )
              goto LABEL_35;
            *v14 = v19;
            --*(_DWORD *)(a3 + 32);
            ++*(_QWORD *)a3;
            break;
          }
        }
      }
LABEL_26:
      sub_14D01A0(v19, a1, a2);
LABEL_19:
      result = ++v18;
      if ( *(_DWORD *)(a2 + 8) <= (int)v18 )
        return result;
    }
    sub_16CCBA0(a3, v19);
    goto LABEL_26;
  }
  return result;
}
