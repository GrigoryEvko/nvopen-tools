// Function: sub_13582D0
// Address: 0x13582d0
//
__int64 __fastcall sub_13582D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rdx
  __int64 v7; // r15
  __int64 v8; // rcx
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rdi
  int v15; // eax
  int v16; // esi
  __int64 v17; // rdi
  int v18; // ecx
  int v19; // esi
  __int64 v20; // rsi
  __int64 v21; // rdi
  int v22; // eax
  int v23; // ecx
  __int64 v24; // rdi
  int v25; // eax
  int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rdi
  int v29; // eax
  int v30; // edx
  int v31; // eax
  int v32; // edx
  __int64 v34; // [rsp+8h] [rbp-48h]
  __int64 v35; // [rsp+10h] [rbp-40h]
  __int64 v36; // [rsp+10h] [rbp-40h]
  __int64 v37; // [rsp+18h] [rbp-38h]
  __int64 v38; // [rsp+18h] [rbp-38h]
  __int64 v39; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 24);
  v3 = *(_QWORD *)(v2 + 32);
  if ( v3 )
  {
    v4 = *(_QWORD *)(v3 + 32);
    if ( v4 )
    {
      v7 = *(_QWORD *)(v4 + 32);
      if ( v7 )
      {
        v8 = *(_QWORD *)(v7 + 32);
        if ( v8 )
        {
          v9 = *(_QWORD *)(v8 + 32);
          if ( v9 )
          {
            v35 = *(_QWORD *)(v7 + 32);
            v37 = *(_QWORD *)(v3 + 32);
            v10 = sub_1357F10(v9, a2);
            v11 = v35;
            v4 = v37;
            v12 = v10;
            v13 = *(_QWORD *)(v35 + 32);
            if ( v12 != v13 )
            {
              *(_DWORD *)(v12 + 64) = (*(_DWORD *)(v12 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v12 + 64) & 0xF8000000;
              v14 = *(_QWORD *)(v35 + 32);
              v15 = *(_DWORD *)(v14 + 64);
              v16 = (v15 + 0x7FFFFFF) & 0x7FFFFFF;
              *(_DWORD *)(v14 + 64) = v16 | v15 & 0xF8000000;
              if ( !v16 )
              {
                v34 = v12;
                sub_1357730(v14, a2);
                v12 = v34;
                v11 = v35;
                v4 = v37;
              }
              *(_QWORD *)(v11 + 32) = v12;
              v13 = v12;
            }
            if ( *(_QWORD *)(v7 + 32) != v13 )
            {
              *(_DWORD *)(v13 + 64) = (*(_DWORD *)(v13 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v13 + 64) & 0xF8000000;
              v17 = *(_QWORD *)(v7 + 32);
              v18 = *(_DWORD *)(v17 + 64);
              v19 = (v18 + 0x7FFFFFF) & 0x7FFFFFF;
              *(_DWORD *)(v17 + 64) = v19 | v18 & 0xF8000000;
              if ( !v19 )
              {
                v36 = v13;
                v38 = v4;
                sub_1357730(v17, a2);
                v13 = v36;
                v4 = v38;
              }
              *(_QWORD *)(v7 + 32) = v13;
            }
            v20 = *(_QWORD *)(v4 + 32);
            v7 = v13;
          }
          else
          {
            v20 = *(_QWORD *)(v4 + 32);
            v7 = *(_QWORD *)(v7 + 32);
          }
          if ( v7 != v20 )
          {
            *(_DWORD *)(v7 + 64) = (*(_DWORD *)(v7 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v7 + 64) & 0xF8000000;
            v21 = *(_QWORD *)(v4 + 32);
            v22 = *(_DWORD *)(v21 + 64);
            v23 = (v22 + 0x7FFFFFF) & 0x7FFFFFF;
            *(_DWORD *)(v21 + 64) = v23 | v22 & 0xF8000000;
            if ( !v23 )
            {
              v39 = v4;
              sub_1357730(v21, a2);
              v4 = v39;
            }
            *(_QWORD *)(v4 + 32) = v7;
          }
          v4 = *(_QWORD *)(v3 + 32);
        }
        if ( v7 != v4 )
        {
          *(_DWORD *)(v7 + 64) = (*(_DWORD *)(v7 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v7 + 64) & 0xF8000000;
          v24 = *(_QWORD *)(v3 + 32);
          v25 = *(_DWORD *)(v24 + 64);
          v26 = (v25 + 0x7FFFFFF) & 0x7FFFFFF;
          *(_DWORD *)(v24 + 64) = v26 | v25 & 0xF8000000;
          if ( !v26 )
            sub_1357730(v24, a2);
          *(_QWORD *)(v3 + 32) = v7;
        }
        v27 = *(_QWORD *)(v2 + 32);
        v3 = v7;
      }
      else
      {
        v27 = *(_QWORD *)(v2 + 32);
        v3 = *(_QWORD *)(v3 + 32);
      }
      if ( v27 != v3 )
      {
        *(_DWORD *)(v3 + 64) = (*(_DWORD *)(v3 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v3 + 64) & 0xF8000000;
        v28 = *(_QWORD *)(v2 + 32);
        v29 = *(_DWORD *)(v28 + 64);
        v30 = (v29 + 0x7FFFFFF) & 0x7FFFFFF;
        *(_DWORD *)(v28 + 64) = v30 | v29 & 0xF8000000;
        if ( !v30 )
          sub_1357730(v28, a2);
        *(_QWORD *)(v2 + 32) = v3;
      }
    }
    *(_QWORD *)(a1 + 24) = v3;
    *(_DWORD *)(v3 + 64) = (*(_DWORD *)(v3 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v3 + 64) & 0xF8000000;
    v31 = *(_DWORD *)(v2 + 64);
    v32 = (v31 + 0x7FFFFFF) & 0x7FFFFFF;
    *(_DWORD *)(v2 + 64) = v32 | v31 & 0xF8000000;
    if ( !v32 )
      sub_1357730(v2, a2);
    return *(_QWORD *)(a1 + 24);
  }
  return v2;
}
