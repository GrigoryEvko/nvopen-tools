// Function: sub_2E646B0
// Address: 0x2e646b0
//
void __fastcall sub_2E646B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  char v3; // dl
  _QWORD **v4; // rax
  _QWORD **v5; // rbx
  _QWORD *v6; // r12
  _QWORD **v7; // r14
  __int64 v8; // r13
  _QWORD **v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rdx

  if ( (!*(_QWORD *)(a1 + 544) || *(_DWORD *)(a1 + 8) == *(_QWORD *)(a1 + 528))
    && (!*(_QWORD *)(a1 + 552) || *(_DWORD *)(a1 + 8) == *(_QWORD *)(a1 + 536)) )
  {
    v2 = *(unsigned int *)(a1 + 588);
    if ( *(_DWORD *)(a1 + 592) == (_DWORD)v2 )
    {
      nullsub_2024();
    }
    else
    {
      v3 = *(_BYTE *)(a1 + 596);
      v4 = *(_QWORD ***)(a1 + 576);
      if ( !v3 )
        v2 = *(unsigned int *)(a1 + 584);
      v5 = &v4[v2];
      if ( v4 == v5 )
      {
LABEL_12:
        v8 = a1 + 568;
      }
      else
      {
        while ( 1 )
        {
          v6 = *v4;
          v7 = v4;
          if ( (unsigned __int64)*v4 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v5 == ++v4 )
            goto LABEL_12;
        }
        v8 = a1 + 568;
        if ( v5 != v4 )
        {
          do
          {
            a2 = (__int64)v6;
            sub_2E64280(a1, (__int64)v6);
            sub_2E32710(v6);
            v9 = v7 + 1;
            if ( v7 + 1 == v5 )
              break;
            while ( 1 )
            {
              v6 = *v9;
              v7 = v9;
              if ( (unsigned __int64)*v9 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v5 == ++v9 )
                goto LABEL_17;
            }
          }
          while ( v5 != v9 );
LABEL_17:
          v3 = *(_BYTE *)(a1 + 596);
        }
      }
      ++*(_QWORD *)(a1 + 568);
      if ( !v3 )
      {
        v10 = 4 * (*(_DWORD *)(a1 + 588) - *(_DWORD *)(a1 + 592));
        v11 = *(unsigned int *)(a1 + 584);
        if ( v10 < 0x20 )
          v10 = 32;
        if ( (unsigned int)v11 > v10 )
        {
          sub_C8C990(v8, a2);
          return;
        }
        memset(*(void **)(a1 + 576), -1, 8 * v11);
      }
      *(_QWORD *)(a1 + 588) = 0;
    }
  }
}
