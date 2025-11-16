// Function: sub_2E790D0
// Address: 0x2e790d0
//
__int64 __fastcall sub_2E790D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r13
  __int64 v7; // r14
  unsigned __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 result; // rax
  unsigned __int64 v11; // r15
  _QWORD *v12; // rax
  _QWORD *v13; // rdx

  v6 = *(_QWORD **)(a2 + 32);
  if ( v6 )
  {
    v7 = *(unsigned __int8 *)(a2 + 43);
    v8 = *(unsigned int *)(a1 + 240);
    if ( (unsigned int)v7 >= (unsigned int)v8 )
    {
      v11 = v7 + 1;
      if ( v7 + 1 != v8 )
      {
        if ( v7 + 1 >= v8 )
        {
          if ( v11 > *(unsigned int *)(a1 + 244) )
          {
            sub_C8D5F0(a1 + 232, (const void *)(a1 + 248), v7 + 1, 8u, a5, a6);
            v8 = *(unsigned int *)(a1 + 240);
          }
          v9 = *(_QWORD *)(a1 + 232);
          v12 = (_QWORD *)(v9 + 8 * v8);
          v13 = (_QWORD *)(v9 + 8 * v11);
          if ( v12 != v13 )
          {
            do
            {
              if ( v12 )
                *v12 = 0;
              ++v12;
            }
            while ( v13 != v12 );
            v9 = *(_QWORD *)(a1 + 232);
          }
          *(_DWORD *)(a1 + 240) = v11;
          goto LABEL_4;
        }
        *(_DWORD *)(a1 + 240) = v11;
      }
    }
    v9 = *(_QWORD *)(a1 + 232);
LABEL_4:
    *v6 = *(_QWORD *)(v9 + 8 * v7);
    *(_QWORD *)(*(_QWORD *)(a1 + 232) + 8 * v7) = v6;
  }
  result = *(_QWORD *)(a1 + 224);
  *(_QWORD *)a2 = result;
  *(_QWORD *)(a1 + 224) = a2;
  return result;
}
