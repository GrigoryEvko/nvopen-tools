// Function: sub_1CCBEA0
// Address: 0x1ccbea0
//
__int64 __fastcall sub_1CCBEA0(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // r15
  __int64 v4; // r12
  unsigned int v5; // ebx
  unsigned int v6; // r13d
  _QWORD *v7; // rax
  unsigned int v8; // edx
  _QWORD *v9; // rcx
  _QWORD *v10; // rsi
  unsigned int v11; // ebx
  unsigned int v12; // eax
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 *v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // rcx
  int v18; // ecx
  int v19; // edi
  int v20; // edx
  int v21; // esi
  unsigned int v22; // [rsp+0h] [rbp-60h]
  int v23; // [rsp+4h] [rbp-5Ch]
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+20h] [rbp-40h]
  unsigned int v28; // [rsp+28h] [rbp-38h]
  unsigned int i; // [rsp+2Ch] [rbp-34h]

  result = (__int64)(a2[1] - *a2) >> 3;
  v23 = result;
  if ( (unsigned int)result > 1 )
  {
    v28 = 0;
    v22 = result - 1;
    do
    {
      result = v28;
      for ( i = v28; v23 != i; result = i )
      {
        v24 = 8LL * i;
        v27 = *(_QWORD *)(*a2 + v24);
        v3 = *(_QWORD *)(v27 + 8);
        v4 = *(_QWORD *)(a1 + 8);
        v5 = *(_DWORD *)(a1 + 24);
        if ( v3 )
        {
          v6 = v5 - 1;
          do
          {
            v7 = sub_1648700(v3);
            if ( *((_BYTE *)v7 + 16) > 0x17u && v5 )
            {
              v8 = v6 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
              v9 = (_QWORD *)(v4 + 8LL * v8);
              v10 = (_QWORD *)*v9;
              if ( v7 == (_QWORD *)*v9 )
              {
LABEL_9:
                if ( (_QWORD *)(v4 + 8LL * v5) != v9 )
                  goto LABEL_17;
              }
              else
              {
                v18 = 1;
                while ( v10 != (_QWORD *)-8LL )
                {
                  v19 = v18 + 1;
                  v8 = v6 & (v18 + v8);
                  v9 = (_QWORD *)(v4 + 8LL * v8);
                  v10 = (_QWORD *)*v9;
                  if ( v7 == (_QWORD *)*v9 )
                    goto LABEL_9;
                  v18 = v19;
                }
              }
            }
            v3 = *(_QWORD *)(v3 + 8);
          }
          while ( v3 );
        }
        if ( v5 )
        {
          v11 = v5 - 1;
          v12 = v11 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v13 = (__int64 *)(v4 + 8LL * v12);
          v14 = *v13;
          if ( v27 == *v13 )
          {
LABEL_13:
            *v13 = -16;
            --*(_DWORD *)(a1 + 16);
            ++*(_DWORD *)(a1 + 20);
          }
          else
          {
            v20 = 1;
            while ( v14 != -8 )
            {
              v21 = v20 + 1;
              v12 = v11 & (v20 + v12);
              v13 = (__int64 *)(v4 + 8LL * v12);
              v14 = *v13;
              if ( v27 == *v13 )
                goto LABEL_13;
              v20 = v21;
            }
          }
        }
        if ( i != v28 )
        {
          v15 = (__int64 *)(*a2 + v24);
          v16 = (__int64 *)(*a2 + 8LL * v28);
          v17 = *v16;
          *v16 = *v15;
          *v15 = v17;
        }
        ++v28;
LABEL_17:
        ++i;
      }
    }
    while ( v22 > v28 );
  }
  return result;
}
