// Function: sub_1EBE4D0
// Address: 0x1ebe4d0
//
void __fastcall sub_1EBE4D0(__int64 a1, int a2, int a3, __int64 a4, int a5, int a6)
{
  unsigned int v6; // edx
  unsigned int v7; // esi
  unsigned int v8; // r14d
  __int64 v9; // rbx
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  _QWORD *v12; // rcx
  _QWORD *v13; // rax

  v6 = a3 & 0x7FFFFFFF;
  if ( *(_DWORD *)(a1 + 928) > v6 )
  {
    v7 = a2 & 0x7FFFFFFF;
    v8 = v7 + 1;
    v9 = v6;
    *(_DWORD *)(*(_QWORD *)(a1 + 920) + 8LL * v6) = 1;
    v10 = *(unsigned int *)(a1 + 928);
    if ( v7 + 1 > (unsigned int)v10 )
    {
      if ( v8 >= v10 )
      {
        if ( v8 > v10 )
        {
          if ( v8 > (unsigned __int64)*(unsigned int *)(a1 + 932) )
          {
            sub_16CD150(a1 + 920, (const void *)(a1 + 936), v8, 8, a5, a6);
            v10 = *(unsigned int *)(a1 + 928);
          }
          v11 = *(_QWORD *)(a1 + 920);
          v12 = (_QWORD *)(v11 + 8LL * v8);
          v13 = (_QWORD *)(v11 + 8 * v10);
          if ( v12 != v13 )
          {
            do
            {
              if ( v13 )
                *v13 = *(_QWORD *)(a1 + 936);
              ++v13;
            }
            while ( v12 != v13 );
            v11 = *(_QWORD *)(a1 + 920);
          }
          *(_DWORD *)(a1 + 928) = v8;
          goto LABEL_4;
        }
      }
      else
      {
        *(_DWORD *)(a1 + 928) = v8;
      }
    }
    v11 = *(_QWORD *)(a1 + 920);
LABEL_4:
    *(_QWORD *)(v11 + 8LL * v7) = *(_QWORD *)(v11 + 8 * v9);
  }
}
