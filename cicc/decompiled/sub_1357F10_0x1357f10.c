// Function: sub_1357F10
// Address: 0x1357f10
//
__int64 __fastcall sub_1357F10(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdi
  int v16; // eax
  int v17; // esi
  __int64 v18; // r8
  int v19; // esi
  int v20; // edi
  __int64 v21; // rsi
  __int64 v22; // rdi
  int v23; // eax
  int v24; // esi
  __int64 v25; // rdi
  int v26; // eax
  int v27; // ecx
  __int64 v28; // rax
  __int64 v29; // rdi
  int v30; // eax
  int v31; // edx
  __int64 v32; // rdi
  int v33; // eax
  int v34; // edx
  __int64 v35; // rax
  __int64 v36; // rdi
  int v37; // eax
  int v38; // edx
  __int64 v40; // [rsp+0h] [rbp-50h]
  __int64 v41; // [rsp+8h] [rbp-48h]
  __int64 v42; // [rsp+8h] [rbp-48h]
  __int64 v43; // [rsp+10h] [rbp-40h]
  __int64 v44; // [rsp+10h] [rbp-40h]
  __int64 v45; // [rsp+10h] [rbp-40h]
  __int64 v46; // [rsp+18h] [rbp-38h]
  __int64 v47; // [rsp+18h] [rbp-38h]
  __int64 v48; // [rsp+18h] [rbp-38h]
  __int64 v49; // [rsp+18h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 32);
  if ( !v3 )
    return a1;
  v4 = *(_QWORD *)(v3 + 32);
  if ( v4 )
  {
    v5 = *(_QWORD *)(v4 + 32);
    if ( v5 )
    {
      v7 = *(_QWORD *)(v5 + 32);
      if ( v7 )
      {
        v8 = *(_QWORD *)(v7 + 32);
        if ( v8 )
        {
          v9 = *(_QWORD *)(v8 + 32);
          if ( v9 )
          {
            if ( *(_QWORD *)(v9 + 32) )
            {
              v41 = *(_QWORD *)(v8 + 32);
              v43 = *(_QWORD *)(v7 + 32);
              v46 = *(_QWORD *)(v5 + 32);
              v10 = sub_1357F10();
              v11 = v41;
              v7 = v46;
              v12 = v10;
              v13 = v43;
              v14 = *(_QWORD *)(v41 + 32);
              if ( v12 != v14 )
              {
                *(_DWORD *)(v12 + 64) = (*(_DWORD *)(v12 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v12 + 64) & 0xF8000000;
                v15 = *(_QWORD *)(v41 + 32);
                v16 = *(_DWORD *)(v15 + 64);
                v17 = (v16 + 0x7FFFFFF) & 0x7FFFFFF;
                *(_DWORD *)(v15 + 64) = v17 | v16 & 0xF8000000;
                if ( !v17 )
                {
                  v40 = v12;
                  sub_1357730(v15, a2);
                  v12 = v40;
                  v11 = v41;
                  v13 = v43;
                  v7 = v46;
                }
                *(_QWORD *)(v11 + 32) = v12;
                v14 = v12;
              }
              if ( *(_QWORD *)(v13 + 32) != v14 )
              {
                *(_DWORD *)(v14 + 64) = (*(_DWORD *)(v14 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v14 + 64) & 0xF8000000;
                v18 = *(_QWORD *)(v13 + 32);
                v19 = *(_DWORD *)(v18 + 64);
                v20 = (v19 + 0x7FFFFFF) & 0x7FFFFFF;
                *(_DWORD *)(v18 + 64) = v20 | v19 & 0xF8000000;
                if ( !v20 )
                {
                  v42 = v14;
                  v44 = v13;
                  v47 = v7;
                  sub_1357730(v18, a2);
                  v14 = v42;
                  v13 = v44;
                  v7 = v47;
                }
                *(_QWORD *)(v13 + 32) = v14;
              }
              v21 = *(_QWORD *)(v7 + 32);
              v8 = v14;
            }
            else
            {
              v21 = *(_QWORD *)(v7 + 32);
              v8 = *(_QWORD *)(v8 + 32);
            }
            if ( v8 != v21 )
            {
              *(_DWORD *)(v8 + 64) = (*(_DWORD *)(v8 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v8 + 64) & 0xF8000000;
              v22 = *(_QWORD *)(v7 + 32);
              v23 = *(_DWORD *)(v22 + 64);
              v24 = (v23 + 0x7FFFFFF) & 0x7FFFFFF;
              *(_DWORD *)(v22 + 64) = v24 | v23 & 0xF8000000;
              if ( !v24 )
              {
                v45 = v8;
                v49 = v7;
                sub_1357730(v22, a2);
                v8 = v45;
                v7 = v49;
              }
              *(_QWORD *)(v7 + 32) = v8;
            }
            v7 = *(_QWORD *)(v5 + 32);
          }
          if ( v7 == v8 )
          {
            v28 = *(_QWORD *)(v4 + 32);
            v5 = v7;
          }
          else
          {
            *(_DWORD *)(v8 + 64) = (*(_DWORD *)(v8 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v8 + 64) & 0xF8000000;
            v25 = *(_QWORD *)(v5 + 32);
            v26 = *(_DWORD *)(v25 + 64);
            v27 = (v26 + 0x7FFFFFF) & 0x7FFFFFF;
            *(_DWORD *)(v25 + 64) = v27 | v26 & 0xF8000000;
            if ( !v27 )
            {
              v48 = v8;
              sub_1357730(v25, a2);
              v8 = v48;
            }
            *(_QWORD *)(v5 + 32) = v8;
            v5 = v8;
            v28 = *(_QWORD *)(v4 + 32);
          }
        }
        else
        {
          v28 = *(_QWORD *)(v4 + 32);
          v5 = *(_QWORD *)(v5 + 32);
        }
        if ( v5 != v28 )
        {
          *(_DWORD *)(v5 + 64) = (*(_DWORD *)(v5 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v5 + 64) & 0xF8000000;
          v29 = *(_QWORD *)(v4 + 32);
          v30 = *(_DWORD *)(v29 + 64);
          v31 = (v30 + 0x7FFFFFF) & 0x7FFFFFF;
          *(_DWORD *)(v29 + 64) = v31 | v30 & 0xF8000000;
          if ( !v31 )
            sub_1357730(v29, a2);
          *(_QWORD *)(v4 + 32) = v5;
        }
        v4 = *(_QWORD *)(v3 + 32);
      }
      if ( v5 != v4 )
      {
        *(_DWORD *)(v5 + 64) = (*(_DWORD *)(v5 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v5 + 64) & 0xF8000000;
        v32 = *(_QWORD *)(v3 + 32);
        v33 = *(_DWORD *)(v32 + 64);
        v34 = (v33 + 0x7FFFFFF) & 0x7FFFFFF;
        *(_DWORD *)(v32 + 64) = v34 | v33 & 0xF8000000;
        if ( !v34 )
          sub_1357730(v32, a2);
        *(_QWORD *)(v3 + 32) = v5;
      }
      v35 = *(_QWORD *)(a1 + 32);
      v3 = v5;
    }
    else
    {
      v35 = *(_QWORD *)(a1 + 32);
      v3 = *(_QWORD *)(v3 + 32);
    }
    if ( v3 != v35 )
    {
      *(_DWORD *)(v3 + 64) = (*(_DWORD *)(v3 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v3 + 64) & 0xF8000000;
      v36 = *(_QWORD *)(a1 + 32);
      v37 = *(_DWORD *)(v36 + 64);
      v38 = (v37 + 0x7FFFFFF) & 0x7FFFFFF;
      *(_DWORD *)(v36 + 64) = v38 | v37 & 0xF8000000;
      if ( !v38 )
        sub_1357730(v36, a2);
      *(_QWORD *)(a1 + 32) = v3;
    }
  }
  return v3;
}
