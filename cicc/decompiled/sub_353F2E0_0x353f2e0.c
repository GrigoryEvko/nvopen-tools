// Function: sub_353F2E0
// Address: 0x353f2e0
//
void __fastcall sub_353F2E0(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // r8
  __int64 v5; // r10
  int *v6; // rbx
  int v7; // edx
  unsigned int v8; // ecx
  int *v9; // r9
  int v10; // r11d
  int *v11; // r13
  unsigned int v12; // eax
  _QWORD *v13; // rbx
  _QWORD *v14; // r13
  __int64 v15; // rax
  __int64 v16; // r14
  _DWORD *v17; // rbx
  int v18; // ecx
  __int64 v19; // rdx
  __int64 v20; // rdx
  unsigned __int64 v21; // r12
  int v22; // r14d
  unsigned int v23; // r15d
  int v24; // r13d
  int v25; // eax
  int v26; // [rsp-3Ch] [rbp-3Ch]

  if ( a3 )
  {
    v4 = *a1;
    if ( a3 - 1 > 0x3FFFFFFE
      || (*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v4 + 8) + 384LL) + 8LL * (a3 >> 6)) & (1LL << a3)) == 0 )
    {
      if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
      {
        v5 = a2 + 16;
        v6 = (int *)(a2 + 80);
        v7 = 15;
      }
      else
      {
        v5 = *(_QWORD *)(a2 + 16);
        v20 = *(unsigned int *)(a2 + 24);
        v6 = (int *)(v5 + 4 * v20);
        if ( !(_DWORD)v20 )
          return;
        v7 = v20 - 1;
      }
      v8 = v7 & (37 * a3);
      v9 = (int *)(v5 + 4LL * v8);
      v10 = *v9;
      v11 = v9;
      if ( a3 == *v9 )
      {
LABEL_6:
        if ( v6 != v11 )
        {
          if ( (*(_BYTE *)(a2 + 8) & 1) != 0 || *(_DWORD *)(a2 + 24) )
          {
            if ( a3 == v10 )
            {
LABEL_10:
              *v9 = -2;
              v12 = *(_DWORD *)(a2 + 8);
              ++*(_DWORD *)(a2 + 12);
              *(_DWORD *)(a2 + 8) = (2 * (v12 >> 1) - 2) | v12 & 1;
              v4 = *a1;
            }
            else
            {
              v25 = 1;
              while ( v10 != -1 )
              {
                v8 = v7 & (v25 + v8);
                v9 = (int *)(v5 + 4LL * v8);
                v10 = *v9;
                if ( a3 == *v9 )
                  goto LABEL_10;
                ++v25;
              }
            }
          }
          v13 = *(_QWORD **)(v4 + 8);
          v14 = (_QWORD *)a1[1];
          v15 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v13 + 16LL) + 200LL))(*(_QWORD *)(*v13 + 16LL));
          v16 = v15;
          if ( (a3 & 0x80000000) != 0 )
          {
            v21 = *(_QWORD *)(v13[7] + 16LL * (a3 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
            v17 = (_DWORD *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v15 + 416LL))(v15, v21);
            v18 = *(_DWORD *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v16 + 376LL))(v16, v21);
          }
          else
          {
            v17 = (_DWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v15 + 424LL))(v15, a3);
            v18 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v16 + 384LL))(v16, a3);
          }
          if ( *v17 == -1 )
            v17 = 0;
          do
          {
            if ( !v17 )
              break;
            v19 = (unsigned int)*v17++;
            *(_DWORD *)(*v14 + 4 * v19) -= v18;
          }
          while ( *v17 != -1 );
        }
      }
      else
      {
        v22 = *v9;
        v23 = v7 & (37 * a3);
        v24 = 1;
        while ( v22 != -1 )
        {
          v23 = v7 & (v24 + v23);
          v26 = v24 + 1;
          v11 = (int *)(v5 + 4LL * v23);
          v22 = *v11;
          if ( a3 == *v11 )
            goto LABEL_6;
          v24 = v26;
        }
      }
    }
  }
}
