// Function: sub_8F0A50
// Address: 0x8f0a50
//
__int64 __fastcall sub_8F0A50(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned int v3; // ecx
  unsigned int v4; // eax
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // r13
  unsigned int *v9; // r14
  void *v10; // r9
  unsigned int *v11; // r11
  _DWORD *v12; // rdi
  __int64 v13; // r10
  unsigned int *v14; // rsi
  unsigned int *v15; // rcx
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r8
  unsigned __int64 v19; // rax
  _DWORD *v20; // r15
  __int64 v21; // rax
  unsigned int v22; // edx
  __int64 v23; // rdx
  unsigned int v24; // [rsp+0h] [rbp-40h]
  __int64 v25; // [rsp+0h] [rbp-40h]
  unsigned int v26; // [rsp+Ch] [rbp-34h]

  v2 = qword_4F690E0;
  *(_DWORD *)(qword_4F690E0 + 2088) = 0;
  v3 = *(_DWORD *)(a1 + 2088);
  qword_4F690E0 = *(_QWORD *)v2;
  if ( v3 )
  {
    v4 = *(_DWORD *)(a2 + 2088);
    if ( v4 )
    {
      v6 = a1;
      if ( v3 >= v4 )
      {
        v22 = v3;
        v3 = *(_DWORD *)(a2 + 2088);
        v4 = v22;
        v23 = a2;
        a2 = a1;
        v6 = v23;
      }
      v7 = a2 + 8;
      v8 = a2 + 8 + 4LL * (int)v4;
      v24 = v3;
      v26 = v3 + v4;
      v9 = (unsigned int *)(v6 + 8);
      v10 = memset((void *)(v2 + 8), 0, 4LL * (int)(v3 + v4));
      if ( a2 + 8 != v8 )
      {
        v11 = &v9[v24];
        v12 = (_DWORD *)(a2 + 8);
        v25 = (int)v24;
        do
        {
          while ( 1 )
          {
            v13 = (unsigned int)*v12;
            if ( *v12 )
            {
              if ( v9 != v11 )
              {
                v14 = (_DWORD *)((char *)v12 + (_QWORD)v10 - v7);
                v15 = v9;
                v16 = 0;
                do
                {
                  v17 = *v15;
                  v18 = *v14;
                  ++v15;
                  ++v14;
                  v19 = v18 + v13 * v17 + v16;
                  *(v14 - 1) = v19;
                  v16 = HIDWORD(v19);
                }
                while ( v11 != v15 );
                v20 = (_DWORD *)((char *)v12 + (_QWORD)v10 - v7 + v25 * 4);
                if ( v16 )
                  break;
              }
            }
            if ( (_DWORD *)v8 == ++v12 )
              goto LABEL_17;
          }
          if ( ((__int64)&v12[v25] - v7) >> 2 == v26 )
          {
            ++v26;
            *v20 = 0;
          }
          ++v12;
          *v20 += v16;
        }
        while ( (_DWORD *)v8 != v12 );
      }
LABEL_17:
      v21 = (int)v26;
      if ( v26 )
      {
        while ( !*(_DWORD *)(v2 + 4 * v21 + 4) )
        {
          if ( !(_DWORD)--v21 )
            goto LABEL_22;
        }
        *(_DWORD *)(v2 + 2088) = v21;
      }
      else
      {
LABEL_22:
        *(_DWORD *)(v2 + 2088) = 0;
      }
    }
  }
  return v2;
}
