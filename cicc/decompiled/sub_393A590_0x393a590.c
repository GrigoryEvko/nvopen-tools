// Function: sub_393A590
// Address: 0x393a590
//
__int64 *__fastcall sub_393A590(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // r14
  _DWORD *v6; // rdx
  __int64 v8; // r15
  _QWORD *v9; // rax
  __int64 v10; // rdx
  _QWORD *v11; // rcx
  void (__fastcall *v12)(_QWORD *); // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // r14
  _QWORD *v15; // r12
  _QWORD *v16; // r15
  unsigned __int64 v17; // rdi
  _QWORD *v18; // r12
  _QWORD *v19; // r15
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // [rsp+8h] [rbp-68h]
  unsigned __int64 v22; // [rsp+10h] [rbp-60h]
  unsigned __int64 *v23; // [rsp+18h] [rbp-58h]
  _QWORD *v24; // [rsp+20h] [rbp-50h]
  _QWORD *v25; // [rsp+20h] [rbp-50h]
  _DWORD *v26; // [rsp+28h] [rbp-48h]
  _QWORD *v27; // [rsp+28h] [rbp-48h]
  unsigned __int64 *v28; // [rsp+28h] [rbp-48h]
  int v29[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v3 = *(_QWORD *)(a2 + 24);
  v4 = *(_QWORD *)(v3 + 8);
  if ( *(_QWORD *)(v3 + 16) - v4 <= 23 )
  {
    *(_DWORD *)(a2 + 8) = 8;
    v29[0] = 8;
    sub_3939040(a1, v29);
  }
  else if ( *(_QWORD *)v4 == 0x8169666F72706CFFLL )
  {
    v5 = *(_QWORD *)(v4 + 8);
    if ( (v5 & 0xFFFFFFFFFFFFFFuLL) > 5 )
    {
      *(_DWORD *)(a2 + 8) = 5;
      v29[0] = 5;
      sub_3939040(a1, v29);
    }
    else
    {
      v6 = sub_393A010(a2, v5, (_DWORD *)(v4 + 40));
      if ( *(_DWORD *)(v4 + 24) )
      {
        *(_DWORD *)(a2 + 8) = 6;
        v29[0] = 6;
        sub_3939040(a1, v29);
      }
      else
      {
        v26 = v6;
        v8 = *(_QWORD *)(v4 + 32);
        v9 = (_QWORD *)sub_22077B0(0x38u);
        if ( v9 )
        {
          v10 = (__int64)v26;
          v27 = v9;
          sub_3939810(v9, (__int64 *)(v4 + v8), v10, v4, 0, v5);
          v9 = v27;
        }
        v11 = *(_QWORD **)(a2 + 32);
        *(_QWORD *)(a2 + 32) = v9;
        v22 = (unsigned __int64)v11;
        if ( v11 )
        {
          v12 = *(void (__fastcall **)(_QWORD *))(*v11 + 8LL);
          if ( v12 == sub_3938740 )
          {
            *v11 = &unk_4A3EF30;
            v13 = v11[1];
            v21 = v13;
            if ( v13 )
            {
              v23 = *(unsigned __int64 **)(v13 + 40);
              v28 = *(unsigned __int64 **)(v13 + 32);
              if ( v23 != v28 )
              {
                do
                {
                  v14 = v28[3];
                  if ( v14 )
                  {
                    v15 = *(_QWORD **)(v14 + 24);
                    v24 = *(_QWORD **)(v14 + 32);
                    if ( v24 != v15 )
                    {
                      do
                      {
                        v16 = (_QWORD *)*v15;
                        while ( v16 != v15 )
                        {
                          v17 = (unsigned __int64)v16;
                          v16 = (_QWORD *)*v16;
                          j_j___libc_free_0(v17);
                        }
                        v15 += 3;
                      }
                      while ( v24 != v15 );
                      v15 = *(_QWORD **)(v14 + 24);
                    }
                    if ( v15 )
                      j_j___libc_free_0((unsigned __int64)v15);
                    v18 = *(_QWORD **)v14;
                    v25 = *(_QWORD **)(v14 + 8);
                    if ( v25 != *(_QWORD **)v14 )
                    {
                      do
                      {
                        v19 = (_QWORD *)*v18;
                        while ( v19 != v18 )
                        {
                          v20 = (unsigned __int64)v19;
                          v19 = (_QWORD *)*v19;
                          j_j___libc_free_0(v20);
                        }
                        v18 += 3;
                      }
                      while ( v25 != v18 );
                      v18 = *(_QWORD **)v14;
                    }
                    if ( v18 )
                      j_j___libc_free_0((unsigned __int64)v18);
                    j_j___libc_free_0(v14);
                  }
                  if ( *v28 )
                    j_j___libc_free_0(*v28);
                  v28 += 7;
                }
                while ( v23 != v28 );
                v28 = *(unsigned __int64 **)(v21 + 32);
              }
              if ( v28 )
                j_j___libc_free_0((unsigned __int64)v28);
              j_j___libc_free_0(v21);
            }
            j_j___libc_free_0(v22);
          }
          else
          {
            v12(v11);
          }
        }
        *(_DWORD *)(a2 + 8) = 0;
        *a1 = 1;
      }
    }
  }
  else
  {
    *(_DWORD *)(a2 + 8) = 3;
    v29[0] = 3;
    sub_3939040(a1, v29);
  }
  return a1;
}
