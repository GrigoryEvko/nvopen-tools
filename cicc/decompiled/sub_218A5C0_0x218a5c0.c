// Function: sub_218A5C0
// Address: 0x218a5c0
//
void __fastcall sub_218A5C0(__int64 a1, unsigned int *a2, __int64 a3)
{
  unsigned int v4; // eax
  _BYTE *v5; // rax
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rbx
  size_t v10; // rax
  void *v11; // rdi
  size_t v12; // r15
  char *src; // [rsp+8h] [rbp-48h]
  char v14; // [rsp+17h] [rbp-39h] BYREF
  char *v15; // [rsp+18h] [rbp-38h] BYREF

  v15 = &v14;
  *(_QWORD *)(__readfsqword(0) - 24) = &v15;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_21892A0;
  if ( &_pthread_key_create )
  {
    v4 = pthread_once(&dword_4FD3940, init_routine);
    if ( !v4 )
    {
      if ( !byte_4FD3930 && (unsigned int)sub_2207590(&byte_4FD3930) )
      {
        qword_4FD3938 = (__int64)&unk_4CD4DA0;
        sub_2207640(&byte_4FD3930);
      }
      v5 = *(_BYTE **)(a3 + 24);
      if ( *(_BYTE **)(a3 + 16) == v5 )
      {
        sub_16E7EE0(a3, "\t", 1u);
      }
      else
      {
        *v5 = 9;
        ++*(_QWORD *)(a3 + 24);
      }
      v6 = *a2;
      v7 = dword_4332F60[v6];
      v8 = dword_4337BE0[v6];
      v9 = v8 | (v7 << 32);
      if ( qword_4FD3938 + (unsigned __int16)v8 != 1 )
      {
        src = (char *)(qword_4FD3938 + (unsigned __int16)v8 - 1);
        v10 = strlen(src);
        v11 = *(void **)(a3 + 24);
        v12 = v10;
        if ( v10 > *(_QWORD *)(a3 + 16) - (_QWORD)v11 )
        {
          sub_16E7EE0(a3, src, v10);
        }
        else if ( v10 )
        {
          memcpy(v11, src, v10);
          *(_QWORD *)(a3 + 24) += v12;
        }
      }
      __asm { jmp     rax }
    }
  }
  else
  {
    v4 = -1;
  }
  sub_4264C5(v4);
}
