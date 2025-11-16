// Function: ._setjmp
// Address: 0x406160
//
// attributes: thunk
int _setjmp(struct __jmp_buf_tag env[1])
{
  return setjmp(env);
}
